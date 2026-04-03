using System.Net.Http.Headers;
using System.Net.Http.Json;
using System.Runtime.CompilerServices;
using System.Text.Json;
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
        var resp = await _client.PostAsJsonAsync("v1/chat/completions", payload,
            OaiCompatJsonContext.Default.OaiRequest, cancellationToken);

        var raw = await resp.Content.ReadAsStringAsync(cancellationToken);
        if (!resp.IsSuccessStatusCode)
            throw new Exception($"[{Name}] API Error ({resp.StatusCode}): {raw}");

        var apiResp = JsonSerializer.Deserialize(raw, OaiCompatJsonContext.Default.OaiResponse);
        if (apiResp?.Choices == null || apiResp.Choices.Length == 0)
            throw new Exception($"[{Name}] Empty response.");

        var choice = apiResp.Choices[0];
        return new CompletionResponse
        {
            Content = choice.Message?.Content ?? string.Empty,
            FinishReason = choice.FinishReason ?? "stop",
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
                ContentDelta = delta.Content,
                FinishReason = chunk.Choices[0].FinishReason
            };
        }
    }

    public Task<float[]> GenerateEmbeddingAsync(
        string input,
        string model = "google/gemini-embedding-001",
        CancellationToken cancellationToken = default)
        => throw new NotSupportedException($"[{Name}] Embeddings not supported on generic OpenAI-compatible backend.");

    public async Task<bool> IsAvailableAsync()
    {
        try
        {
            var resp = await _client.GetAsync("v1/models");
            return resp.IsSuccessStatusCode;
        }
        catch { return false; }
    }

    private OaiRequest BuildPayload(CompletionRequest request, bool stream) => new()
    {
        Model = _model ?? request.Model,
        Messages = request.Messages.Select(m => new OaiMessage
        {
            Role = m.Role,
            Content = m.Content
        }).ToArray(),
        MaxTokens = request.MaxTokens,
        Temperature = request.Temperature,
        Stream = stream ? true : null
    };
}

// ── Internal AOT models ────────────────────────────────────────────────────────

internal sealed class OaiRequest
{
    [JsonPropertyName("model")] public required string Model { get; init; }
    [JsonPropertyName("messages")] public required OaiMessage[] Messages { get; init; }
    [JsonPropertyName("max_tokens")] public int MaxTokens { get; init; }
    [JsonPropertyName("temperature")] public double Temperature { get; init; }
    [JsonPropertyName("stream")] public bool? Stream { get; init; }
}

internal sealed class OaiMessage
{
    [JsonPropertyName("role")] public required string Role { get; init; }
    [JsonPropertyName("content")] public required string Content { get; init; }
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

[JsonSourceGenerationOptions(DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull)]
[JsonSerializable(typeof(OaiRequest))]
[JsonSerializable(typeof(OaiResponse))]
[JsonSerializable(typeof(OaiStreamResponse))]
internal partial class OaiCompatJsonContext : JsonSerializerContext { }
