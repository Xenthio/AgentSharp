using System.Net.Http.Headers;
using System.Net.Http.Json;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization;
using AgentSharp.Core.Interfaces;

namespace AgentSharp.Core.Backends;

/// <summary>
/// Generic OpenAI-compatible chat-completions backend.
/// Used as the underlying impl for Oobabooga/LmStudio chat mode,
/// and directly usable for any OpenAI-compatible server.
/// </summary>
public sealed class OpenAiCompatibleBackend : IBackendProvider
{
    public string Name => "openai-compatible";

    private readonly HttpClient _client;
    private readonly string _baseUrl;
    private readonly string? _model;

    public OpenAiCompatibleBackend(string baseUrl, string? apiKey = null, string? model = null)
    {
        _baseUrl = baseUrl.TrimEnd('/');
        _model = model;
        _client = new HttpClient
        {
            BaseAddress = new Uri(_baseUrl + "/"),
            Timeout = TimeSpan.FromMinutes(5)
        };
        if (!string.IsNullOrWhiteSpace(apiKey))
            _client.DefaultRequestHeaders.Authorization =
                new AuthenticationHeaderValue("Bearer", apiKey);
    }

    public async Task<CompletionResponse> GenerateCompletionAsync(
        CompletionRequest request,
        CancellationToken cancellationToken = default)
    {
        var payload = BuildPayload(request, stream: false);
        var payloadJson = JsonSerializer.Serialize(payload, OaiCompatJsonContext.Default.OaiRequest);
        // Uncomment to debug sampling params: Console.WriteLine($"[{Name}] Request payload: {payloadJson}");
        
        var resp = await _client.PostAsJsonAsync("v1/chat/completions", payload,
            OaiCompatJsonContext.Default.OaiRequest, cancellationToken);

        var raw = await resp.Content.ReadAsStringAsync(cancellationToken);
        if (!resp.IsSuccessStatusCode)
            throw new Exception($"[{Name}] API Error ({resp.StatusCode}): {raw}");

        var apiResp = JsonSerializer.Deserialize(raw, OaiCompatJsonContext.Default.OaiResponse);
        if (apiResp?.Choices == null || apiResp.Choices.Length == 0)
            throw new Exception($"[{Name}] Empty response.");

        var choice = apiResp.Choices[0];
        
        // Extract tool calls if present
        List<AgentSharp.Core.Interfaces.ToolCall>? toolCalls = null;
        if (choice.Message?.ToolCalls != null && choice.Message.ToolCalls.Length > 0)
        {
            toolCalls = choice.Message.ToolCalls.Select(tc => new AgentSharp.Core.Interfaces.ToolCall
            {
                Id = tc.Id ?? string.Empty,
                Type = tc.Type ?? "function",
                Function = new AgentSharp.Core.Interfaces.ToolCallFunction
                {
                    Name = tc.Function?.Name ?? string.Empty,
                    Arguments = tc.Function?.Arguments ?? "{}"
                }
            }).ToList();
        }
        
        return new CompletionResponse
        {
            Content = choice.Message?.Content?.ToString() ?? string.Empty,
            ReasoningContent = choice.Message?.ReasoningContent,
            FinishReason = choice.FinishReason ?? "stop",
            ToolCalls = toolCalls,
            Usage = new TokenUsage
            {
                PromptTokens = apiResp.Usage?.PromptTokens ?? 0,
                CompletionTokens = apiResp.Usage?.CompletionTokens ?? 0,
                TotalTokens = (apiResp.Usage?.PromptTokens ?? 0) + (apiResp.Usage?.CompletionTokens ?? 0)
            }
        };
    }

    public async IAsyncEnumerable<CompletionStreamChunk> GenerateCompletionStreamAsync(
        CompletionRequest request,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var payload = BuildPayload(request, stream: true);
        var payloadJson = JsonSerializer.Serialize(payload, OaiCompatJsonContext.Default.OaiRequest);
        // Uncomment to debug sampling params: Console.WriteLine($"[{Name}] Request payload: {payloadJson}");
        
        var reqMsg = new HttpRequestMessage(HttpMethod.Post, "v1/chat/completions")
        {
            Content = JsonContent.Create(payload, OaiCompatJsonContext.Default.OaiRequest)
        };

        using var resp = await _client.SendAsync(reqMsg, HttpCompletionOption.ResponseHeadersRead, cancellationToken);
        if (!resp.IsSuccessStatusCode)
        {
            var err = await resp.Content.ReadAsStringAsync(cancellationToken);
            throw new Exception($"[{Name}] API Error ({resp.StatusCode}): {err}");
        }

        using var stream = await resp.Content.ReadAsStreamAsync(cancellationToken);
        using var reader = new StreamReader(stream);

        while (!reader.EndOfStream && !cancellationToken.IsCancellationRequested)
        {
            var line = await reader.ReadLineAsync(cancellationToken);
            if (string.IsNullOrWhiteSpace(line)) continue;
            if (line.StartsWith("data: [DONE]")) break;
            if (!line.StartsWith("data: ")) continue;

            var json = line["data: ".Length..];
            OaiStreamResponse? chunk;
            try { chunk = JsonSerializer.Deserialize(json, OaiCompatJsonContext.Default.OaiStreamResponse); }
            catch { continue; }

            if (chunk?.Choices == null || chunk.Choices.Length == 0) continue;
            var delta = chunk.Choices[0].Delta;
            if (delta == null) continue;

            yield return new CompletionStreamChunk
            {
                ContentDelta = delta.Content?.ToString(),
                FinishReason = chunk.Choices[0].FinishReason
            };
        }
    }

    public async Task<float[]> GenerateEmbeddingAsync(
        string input,
        string model = "nomic-embed-text",
        CancellationToken cancellationToken = default)
    {
        var req = new OaiEmbeddingRequest { Model = model, Input = input };
        var resp = await _client.PostAsJsonAsync("v1/embeddings", req,
            OaiCompatJsonContext.Default.OaiEmbeddingRequest, cancellationToken);

        var raw = await resp.Content.ReadAsStringAsync(cancellationToken);
        if (!resp.IsSuccessStatusCode)
            throw new Exception($"[{Name}] Embedding API Error ({resp.StatusCode}): {raw}");

        var result = JsonSerializer.Deserialize(raw, OaiCompatJsonContext.Default.OaiEmbeddingResponse);
        if (result?.Data == null || result.Data.Length == 0 || result.Data[0].Embedding == null)
            throw new InvalidOperationException($"[{Name}] Embedding response missing data.");

        return result.Data[0].Embedding!;
    }

    public Task<int> CountTokensAsync(string text, CancellationToken cancellationToken = default)
    {
        // LM Studio doesn't expose a tokenize endpoint via REST API.
        // Use character-based estimation: ~3.5 chars per token for English text.
        // This is more accurate than the default 4:1 ratio for most LLM tokenizers.
        return Task.FromResult(text.Length * 2 / 7); // ~3.5 chars per token
    }

    public async Task<bool> IsAvailableAsync()
    {
        try
        {
            var resp = await _client.GetAsync("v1/models");
            return resp.IsSuccessStatusCode;
        }
        catch { return false; }
    }

    /// <summary>
    /// LM Studio requires plain base64 without "data:mime;base64," prefix.
    /// Strip it if present so we work with both LM Studio and standard OpenAI.
    /// </summary>
    private static string StripDataPrefix(string url)
    {
        if (url.StartsWith("data:") && url.Contains(";base64,"))
        {
            var idx = url.IndexOf(";base64,");
            return url[(idx + 8)..]; // everything after ";base64,"
        }
        return url;
    }

    private OaiRequest BuildPayload(CompletionRequest request, bool stream) => new()
    {
        Model = _model ?? request.Model,
        Messages = request.Messages.Select(m =>
        {
            JsonNode? content;
            if (m.ContentParts != null)
            {
                var arr = new JsonArray();
                foreach (var part in m.ContentParts)
                {
                    if (part.Type == "text")
                        arr.Add(new JsonObject { ["type"] = "text", ["text"] = part.Text });
                    else if (part.Type == "image_url" && part.ImageUrl != null)
                    {
                        // Gemma 4 and most vision models require full data: URL format
                        // e.g. data:image/png;base64,iVBORw0K...
                        var url = part.ImageUrl.Url;
                        Console.WriteLine($"[Vision] Sending image, url starts with: {url[..Math.Min(40, url.Length)]}...");
                        var imgObj = new JsonObject { ["url"] = url };
                        if (part.ImageUrl.Detail != "auto")
                            imgObj["detail"] = part.ImageUrl.Detail;
                        arr.Add(new JsonObject
                        {
                            ["type"]      = "image_url",
                            ["image_url"] = imgObj
                        });
                    }
                }
                content = arr;
            }
            else
            {
                content = m.Content;
            }
            var oaiMsg = new OaiMessage { Role = m.Role, Content = content, ToolCallId = m.ToolCallId, Name = m.Name };
            // Include tool_calls for assistant messages that have them
            if (m.ToolCalls != null && m.ToolCalls.Count > 0)
            {
                oaiMsg = new OaiMessage
                {
                    Role = m.Role,
                    Content = content,
                    ToolCalls = m.ToolCalls.Select(tc => new OaiToolCall
                    {
                        Id = tc.Id,
                        Type = tc.Type,
                        Function = new OaiToolCallFunction { Name = tc.Function.Name, Arguments = tc.Function.Arguments }
                    }).ToArray()
                };
            }
            return oaiMsg;
        }).ToArray(),
        MaxTokens   = request.MaxTokens,
        Temperature = request.Temperature,
        Stream      = stream ? true : null,
        MinP          = request.MinP > 0 ? request.MinP : null,
        TopK          = request.TopK > 0 ? request.TopK : null,
        TopP          = request.TopP < 1.0 ? request.TopP : null,
        RepeatPenalty = request.RepeatPenalty != 1.0 ? request.RepeatPenalty : null,
        Seed          = request.Seed >= 0 ? request.Seed : null,
        // LM Studio reasoning field (takes precedence)
        Reasoning      = !string.IsNullOrEmpty(request.ReasoningEffort) ? request.ReasoningEffort : null,
        // Legacy: Anthropic-style thinking block (Claude)
        Thinking      = (!string.IsNullOrEmpty(request.ReasoningEffort) || !request.EnableThinking) ? null
            : new JsonObject { ["type"] = "enabled", ["budget_tokens"] = request.ThinkingBudget > 0 ? request.ThinkingBudget : 2048 },
        // Legacy: Qwen 3 enable_thinking bool
        EnableThinking = !string.IsNullOrEmpty(request.ReasoningEffort) ? null
            : request.EnableThinking ? null : (bool?)false,
        LogitBias      = request.LogitBias != null && request.LogitBias.Count > 0 ? request.LogitBias : null,
        Tools          = request.Tools?.Select(t => System.Text.Json.Nodes.JsonNode.Parse(JsonSerializer.Serialize(t))).ToArray(),
        ToolChoice     = request.Tools?.Count > 0 ? "auto" : null,
    };
}

// ── Internal AOT models ────────────────────────────────────────────────────────

internal sealed class OaiRequest
{
    [JsonPropertyName("model")]       public required string Model { get; init; }
    [JsonPropertyName("messages")]    public required OaiMessage[] Messages { get; init; }
    [JsonPropertyName("max_tokens")]  public int MaxTokens { get; init; }
    [JsonPropertyName("temperature")] public double Temperature { get; init; }
    [JsonPropertyName("stream")]      public bool? Stream { get; init; }
    [JsonPropertyName("min_p")]           public double? MinP { get; init; }
    [JsonPropertyName("top_k")]           public int? TopK { get; init; }
    [JsonPropertyName("top_p")]           public double? TopP { get; init; }
    [JsonPropertyName("repeat_penalty")]  public double? RepeatPenalty { get; init; }
    [JsonPropertyName("seed")]            public int? Seed { get; init; }
    [JsonPropertyName("thinking")]        public JsonNode? Thinking { get; init; }
    [JsonPropertyName("enable_thinking")] public bool? EnableThinking { get; init; }
    /// <summary>LM Studio reasoning setting: "off" | "low" | "medium" | "high" | "on"</summary>
    [JsonPropertyName("reasoning")]       public string? Reasoning { get; init; }
    [JsonPropertyName("logit_bias")]       public Dictionary<string, float>? LogitBias { get; init; }
    [JsonPropertyName("tools")]            public System.Text.Json.Nodes.JsonNode[]? Tools { get; init; }
    [JsonPropertyName("tool_choice")]      public string? ToolChoice { get; init; }
}

internal sealed class OaiMessage
{
    [JsonPropertyName("role")] public required string Role { get; init; }
    [JsonPropertyName("content")] public required JsonNode? Content { get; init; }
    /// <summary>reasoning_content field returned by Qwen3, DeepSeek R1, etc.</summary>
    [JsonPropertyName("reasoning_content")] public string? ReasoningContent { get; init; }
    /// <summary>tool_calls from OpenAI-compatible APIs (response)</summary>
    [JsonPropertyName("tool_calls")] public OaiToolCall[]? ToolCalls { get; init; }
    /// <summary>tool_call_id for tool result messages</summary>
    [JsonPropertyName("tool_call_id")] public string? ToolCallId { get; init; }
    /// <summary>Name for tool result messages</summary>
    [JsonPropertyName("name")] public string? Name { get; init; }
}

internal sealed class OaiResponse
{
    [JsonPropertyName("choices")] public OaiChoice[]? Choices { get; init; }
    [JsonPropertyName("usage")] public OaiUsage? Usage { get; init; }
}

internal sealed class OaiChoice
{
    [JsonPropertyName("message")] public OaiMessage? Message { get; init; }
    [JsonPropertyName("finish_reason")] public string? FinishReason { get; init; }
}

internal sealed class OaiStreamResponse
{
    [JsonPropertyName("choices")] public OaiStreamChoice[]? Choices { get; init; }
}

internal sealed class OaiStreamChoice
{
    [JsonPropertyName("delta")] public OaiMessage? Delta { get; init; }
    [JsonPropertyName("finish_reason")] public string? FinishReason { get; init; }
}

internal sealed class OaiUsage
{
    [JsonPropertyName("prompt_tokens")] public int PromptTokens { get; init; }
    [JsonPropertyName("completion_tokens")] public int CompletionTokens { get; init; }
}

internal sealed class OaiEmbeddingRequest
{
    [JsonPropertyName("model")] public required string Model { get; init; }
    [JsonPropertyName("input")] public required string Input { get; init; }
}

internal sealed class OaiEmbeddingResponse
{
    [JsonPropertyName("data")] public OaiEmbeddingData[]? Data { get; init; }
}

internal sealed class OaiEmbeddingData
{
    [JsonPropertyName("embedding")] public float[]? Embedding { get; init; }
}

internal sealed class OaiToolCall
{
    [JsonPropertyName("id")] public string? Id { get; init; }
    [JsonPropertyName("type")] public string? Type { get; init; }
    [JsonPropertyName("function")] public OaiToolCallFunction? Function { get; init; }
}

internal sealed class OaiToolCallFunction
{
    [JsonPropertyName("name")] public string? Name { get; init; }
    [JsonPropertyName("arguments")] public string? Arguments { get; init; }
}

[JsonSourceGenerationOptions(DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull)]
[JsonSerializable(typeof(OaiRequest))]
[JsonSerializable(typeof(OaiResponse))]
[JsonSerializable(typeof(OaiStreamResponse))]
[JsonSerializable(typeof(OaiEmbeddingRequest))]
[JsonSerializable(typeof(OaiEmbeddingResponse))]
[JsonSerializable(typeof(OaiToolCall))]
[JsonSerializable(typeof(OaiToolCallFunction))]
[JsonSerializable(typeof(AgentSharp.Core.Interfaces.ToolDefinition))]
[JsonSerializable(typeof(AgentSharp.Core.Interfaces.FunctionDefinition))]
[JsonSerializable(typeof(System.Text.Json.Nodes.JsonObject))]
[JsonSerializable(typeof(System.Text.Json.Nodes.JsonArray))]
[JsonSerializable(typeof(System.Text.Json.Nodes.JsonNode))]
[JsonSerializable(typeof(System.Text.Json.Nodes.JsonValue))]
internal partial class OaiCompatJsonContext : JsonSerializerContext { }
