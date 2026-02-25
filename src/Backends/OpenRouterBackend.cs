using System.Net.Http.Json;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Serialization;
using AgentSharp.Core.Interfaces;

namespace AgentSharp.Core.Backends;

/// <summary>
/// OpenRouter backend implementation
/// </summary>
public sealed class OpenRouterBackend : IBackendProvider
{
    private readonly HttpClient _client;
    private readonly string _apiKey;
    private readonly string _baseUrl;

    public OpenRouterBackend(string apiKey, string? baseUrl = null)
    {
        _apiKey = apiKey ?? throw new ArgumentNullException(nameof(apiKey));
        _baseUrl = baseUrl ?? "https://openrouter.ai/api/v1";
        
        if (!_baseUrl.EndsWith("/"))
        {
            _baseUrl += "/";
        }
        
        _client = new HttpClient
        {
            BaseAddress = new Uri(_baseUrl),
            Timeout = TimeSpan.FromMinutes(5)
        };
        _client.DefaultRequestHeaders.Add("Authorization", $"Bearer {_apiKey}");
        _client.DefaultRequestHeaders.Add("HTTP-Referer", "https://github.com/agentsharp");
        _client.DefaultRequestHeaders.Add("X-Title", "AgentSharp");
    }

    public string Name => "openrouter";

    public async Task<CompletionResponse> GenerateCompletionAsync(
        CompletionRequest request,
        CancellationToken cancellationToken = default)
    {
        var apiRequest = new OpenRouterRequest
        {
            Model = request.Model,
            Messages = request.Messages.Select(ConvertMessage).ToArray(),
            Temperature = request.Temperature,
            MaxTokens = request.MaxTokens,
            Tools = request.Tools?.Select(ConvertTool).ToArray()
        };

        var response = await _client.PostAsJsonAsync("chat/completions", apiRequest, OpenRouterJsonContext.Default.OpenRouterRequest, cancellationToken);
        var rawContent = await response.Content.ReadAsStringAsync(cancellationToken);

        if (!response.IsSuccessStatusCode)
        {
            throw new Exception($"OpenRouter API Error ({response.StatusCode}): {rawContent}");
        }

        var apiResponse = JsonSerializer.Deserialize(rawContent, OpenRouterJsonContext.Default.OpenRouterResponse);
        if (apiResponse?.Choices == null || apiResponse.Choices.Length == 0)
        {
            throw new Exception("OpenRouter returned an empty response.");
        }

        var choice = apiResponse.Choices[0];
        
        var toolCalls = choice.Message.ToolCalls?.Select(tc => new AgentSharp.Core.Interfaces.ToolCall
        {
            Id = tc.Id,
            Type = tc.Type,
            Function = new AgentSharp.Core.Interfaces.ToolCallFunction
            {
                Name = tc.Function.Name,
                Arguments = tc.Function.Arguments ?? "{}"
            }
        }).ToList();

        return new CompletionResponse
        {
            Content = choice.Message.Content ?? string.Empty,
            ToolCalls = toolCalls,
            Usage = new TokenUsage
            {
                PromptTokens = apiResponse.Usage?.PromptTokens ?? 0,
                CompletionTokens = apiResponse.Usage?.CompletionTokens ?? 0,
                TotalTokens = (apiResponse.Usage?.PromptTokens ?? 0) + (apiResponse.Usage?.CompletionTokens ?? 0)
            },
            FinishReason = choice.FinishReason ?? "stop"
        };
    }

    public async IAsyncEnumerable<CompletionStreamChunk> GenerateCompletionStreamAsync(
        CompletionRequest request,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var apiRequest = new OpenRouterRequest
        {
            Model = request.Model,
            Messages = request.Messages.Select(ConvertMessage).ToArray(),
            Temperature = request.Temperature,
            MaxTokens = request.MaxTokens,
            Tools = request.Tools?.Select(ConvertTool).ToArray(),
            Stream = true,
            IncludeReasoning = true
        };

        var requestMsg = new HttpRequestMessage(HttpMethod.Post, "chat/completions")
        {
            Content = JsonContent.Create(apiRequest, OpenRouterJsonContext.Default.OpenRouterRequest)
        };

        using var response = await _client.SendAsync(requestMsg, HttpCompletionOption.ResponseHeadersRead, cancellationToken);
        
        if (!response.IsSuccessStatusCode)
        {
            var rawError = await response.Content.ReadAsStringAsync(cancellationToken);
            throw new Exception($"OpenRouter API Error ({response.StatusCode}): {rawError}");
        }

        using var stream = await response.Content.ReadAsStreamAsync(cancellationToken);
        using var reader = new StreamReader(stream);

        var activeToolCalls = new Dictionary<int, AgentSharp.Core.Interfaces.ToolCall>();

        while (!reader.EndOfStream && !cancellationToken.IsCancellationRequested)
        {
            var line = await reader.ReadLineAsync(cancellationToken);
            if (string.IsNullOrWhiteSpace(line)) continue;
            if (line.StartsWith("data: [DONE]")) break;
            if (!line.StartsWith("data: ")) continue;

            var json = line.Substring("data: ".Length);
            var apiResponse = JsonSerializer.Deserialize(json, OpenRouterJsonContext.Default.OpenRouterStreamResponse);

            if (apiResponse?.Choices == null || apiResponse.Choices.Length == 0) continue;

            var choice = apiResponse.Choices[0];
            var delta = choice.Delta;

            if (delta == null) continue;

            // Handle Tool Call streaming by assembling fragments
            if (delta.ToolCalls != null)
            {
                foreach (var tcDelta in delta.ToolCalls)
                {
                    int index = tcDelta.Index ?? 0;
                    if (!activeToolCalls.TryGetValue(index, out var currentTc))
                    {
                        currentTc = new AgentSharp.Core.Interfaces.ToolCall
                        {
                            Id = tcDelta.Id ?? "",
                            Type = tcDelta.Type ?? "function",
                            Function = new AgentSharp.Core.Interfaces.ToolCallFunction
                            {
                                Name = tcDelta.Function?.Name ?? "",
                                Arguments = tcDelta.Function?.Arguments ?? ""
                            }
                        };
                        activeToolCalls[index] = currentTc;
                    }
                    else
                    {
                        if (!string.IsNullOrEmpty(tcDelta.Function?.Arguments))
                        {
                            currentTc.Function.Arguments += tcDelta.Function.Arguments;
                        }
                    }
                }
            }

            // Return the chunk
            yield return new CompletionStreamChunk
            {
                ContentDelta = delta.Content,
                ReasoningDelta = delta.Reasoning,
                FinishReason = choice.FinishReason,
                // Only return tool calls when the stream finishes them
                ToolCalls = choice.FinishReason == "tool_calls" || choice.FinishReason == "stop" ? activeToolCalls.Values.ToList() : null
            };
        }
    }

    private static OpenRouterMessage ConvertMessage(ChatMessage msg)
    {
        return new OpenRouterMessage
        {
            Role = msg.Role,
            Content = msg.Content,
            Name = msg.Name,
            ToolCallId = msg.ToolCallId,
            ToolCalls = msg.ToolCalls?.Select(tc => new OpenRouterToolCall
            {
                Id = tc.Id,
                Type = tc.Type,
                Function = new OpenRouterFunction
                {
                    Name = tc.Function.Name,
                    Arguments = tc.Function.Arguments
                }
            }).ToArray()
        };
    }

    private static OpenRouterTool ConvertTool(ToolDefinition tool) => new()
    {
        Type = tool.Type,
        Function = new OpenRouterFunctionDef
        {
            Name = tool.Function.Name,
            Description = tool.Function.Description,
            Parameters = tool.Function.Parameters
        }
    };
}

// OpenRouter API models (AOT-friendly with source generation)

internal sealed class OpenRouterRequest
{
    [JsonPropertyName("model")]
    public required string Model { get; init; }

    [JsonPropertyName("messages")]
    public required OpenRouterMessage[] Messages { get; init; }

    [JsonPropertyName("tools")]
    public OpenRouterTool[]? Tools { get; init; }

    [JsonPropertyName("temperature")]
    public double? Temperature { get; init; }

    [JsonPropertyName("max_tokens")]
    public int? MaxTokens { get; init; }

    [JsonPropertyName("stream")]
    public bool? Stream { get; init; }

    [JsonPropertyName("include_reasoning")]
    public bool? IncludeReasoning { get; init; }
}

internal sealed class OpenRouterResponse
{
    [JsonPropertyName("id")]
    public string? Id { get; init; }

    [JsonPropertyName("choices")]
    public required OpenRouterChoice[] Choices { get; init; }

    [JsonPropertyName("usage")]
    public OpenRouterUsage? Usage { get; init; }
}

internal sealed class OpenRouterChoice
{
    [JsonPropertyName("message")]
    public required OpenRouterMessage Message { get; init; }

    [JsonPropertyName("finish_reason")]
    public string? FinishReason { get; init; }
}

internal sealed class OpenRouterStreamResponse
{
    [JsonPropertyName("id")]
    public string? Id { get; init; }

    [JsonPropertyName("choices")]
    public OpenRouterStreamChoice[]? Choices { get; init; }

    [JsonPropertyName("usage")]
    public OpenRouterUsage? Usage { get; init; }
}

internal sealed class OpenRouterStreamChoice
{
    [JsonPropertyName("delta")]
    public OpenRouterStreamDelta? Delta { get; init; }

    [JsonPropertyName("finish_reason")]
    public string? FinishReason { get; init; }
}

internal sealed class OpenRouterStreamDelta
{
    [JsonPropertyName("role")]
    public string? Role { get; init; }

    [JsonPropertyName("content")]
    public string? Content { get; init; }

    [JsonPropertyName("reasoning")]
    public string? Reasoning { get; init; }

    [JsonPropertyName("tool_calls")]
    public OpenRouterStreamToolCall[]? ToolCalls { get; init; }
}

internal sealed class OpenRouterStreamToolCall
{
    [JsonPropertyName("index")]
    public int? Index { get; init; }

    [JsonPropertyName("id")]
    public string? Id { get; init; }

    [JsonPropertyName("type")]
    public string? Type { get; init; }

    [JsonPropertyName("function")]
    public OpenRouterStreamFunction? Function { get; init; }
}

internal sealed class OpenRouterStreamFunction
{
    [JsonPropertyName("name")]
    public string? Name { get; init; }

    [JsonPropertyName("arguments")]
    public string? Arguments { get; init; }
}

internal sealed class OpenRouterMessage
{
    [JsonPropertyName("role")]
    public string? Role { get; init; }

    [JsonPropertyName("content")]
    public string? Content { get; init; }

    [JsonPropertyName("name")]
    public string? Name { get; init; }

    [JsonPropertyName("tool_calls")]
    public OpenRouterToolCall[]? ToolCalls { get; init; }

    [JsonPropertyName("tool_call_id")]
    public string? ToolCallId { get; init; }
}

internal sealed class OpenRouterToolCall
{
    [JsonPropertyName("id")]
    public required string Id { get; init; }

    [JsonPropertyName("type")]
    public required string Type { get; init; }

    [JsonPropertyName("function")]
    public required OpenRouterFunction Function { get; init; }
}

internal sealed class OpenRouterFunction
{
    [JsonPropertyName("name")]
    public required string Name { get; init; }

    [JsonPropertyName("arguments")]
    public required string Arguments { get; init; }
}

internal sealed class OpenRouterTool
{
    [JsonPropertyName("type")]
    public required string Type { get; init; }

    [JsonPropertyName("function")]
    public required OpenRouterFunctionDef Function { get; init; }
}

internal sealed class OpenRouterFunctionDef
{
    [JsonPropertyName("name")]
    public required string Name { get; init; }

    [JsonPropertyName("description")]
    public required string Description { get; init; }

    [JsonPropertyName("parameters")]
    public required object Parameters { get; init; }
}

internal sealed class OpenRouterUsage
{
    [JsonPropertyName("prompt_tokens")]
    public int PromptTokens { get; init; }

    [JsonPropertyName("completion_tokens")]
    public int CompletionTokens { get; init; }
}

// AOT-friendly JSON source generation
[JsonSourceGenerationOptions(
    PropertyNamingPolicy = JsonKnownNamingPolicy.SnakeCaseLower,
    DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull)]
[JsonSerializable(typeof(OpenRouterRequest))]
[JsonSerializable(typeof(OpenRouterResponse))]
[JsonSerializable(typeof(OpenRouterStreamResponse))]
[JsonSerializable(typeof(ToolDefinition))]
[JsonSerializable(typeof(System.Text.Json.Nodes.JsonObject))]
internal partial class OpenRouterJsonContext : JsonSerializerContext
{
}
