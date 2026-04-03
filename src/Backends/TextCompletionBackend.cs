using System.Net.Http.Json;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Serialization;
using AgentSharp.Core.Interfaces;

namespace AgentSharp.Core.Backends;

/// <summary>
/// OpenAI-compatible text-completions backend (POST /v1/completions).
/// Used as fallback impl for Oobabooga/LmStudio when chat API is unavailable.
/// Converts the chat message list into a flat text prompt.
/// </summary>
public sealed class TextCompletionBackend : IBackendProvider
{
    public string Name => "text-completion";

    private readonly HttpClient _client;
    private readonly string _baseUrl;

    public TextCompletionBackend(string baseUrl)
    {
        _baseUrl = baseUrl.TrimEnd('/');
        _client = new HttpClient
        {
            BaseAddress = new Uri(_baseUrl + "/"),
            Timeout = TimeSpan.FromMinutes(5)
        };
    }

    public async Task<CompletionResponse> GenerateCompletionAsync(
        CompletionRequest request,
        CancellationToken cancellationToken = default)
    {
        var payload = BuildPayload(request, stream: false);
        var resp = await _client.PostAsJsonAsync("v1/completions", payload,
            TextCompletionJsonContext.Default.TcRequest, cancellationToken);

        var raw = await resp.Content.ReadAsStringAsync(cancellationToken);
        if (!resp.IsSuccessStatusCode)
            throw new Exception($"[{Name}] API Error ({resp.StatusCode}): {raw}");

        var apiResp = JsonSerializer.Deserialize(raw, TextCompletionJsonContext.Default.TcResponse);
        if (apiResp?.Choices == null || apiResp.Choices.Length == 0)
            throw new Exception($"[{Name}] Empty response.");

        var text = apiResp.Choices[0].Text ?? string.Empty;
        return new CompletionResponse
        {
            Content = text.Trim(),
            FinishReason = apiResp.Choices[0].FinishReason ?? "stop",
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
        var reqMsg = new HttpRequestMessage(HttpMethod.Post, "v1/completions")
        {
            Content = JsonContent.Create(payload, TextCompletionJsonContext.Default.TcRequest)
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
            TcStreamResponse? chunk;
            try { chunk = JsonSerializer.Deserialize(json, TextCompletionJsonContext.Default.TcStreamResponse); }
            catch { continue; }

            if (chunk?.Choices == null || chunk.Choices.Length == 0) continue;
            var text = chunk.Choices[0].Text;
            if (string.IsNullOrEmpty(text)) continue;

            yield return new CompletionStreamChunk
            {
                ContentDelta = text,
                FinishReason = chunk.Choices[0].FinishReason
            };
        }
    }

    public Task<float[]> GenerateEmbeddingAsync(
        string input,
        string model = "google/gemini-embedding-001",
        CancellationToken cancellationToken = default)
        => throw new NotSupportedException($"[{Name}] Embeddings not supported on text-completion backend.");

    /// <summary>
    /// Builds a flat text prompt from chat messages.
    /// System message is prepended as-is; then [Author]: Content lines follow.
    /// Images in ContentParts are ignored (not supported in text-completion mode).
    /// </summary>
    private static string BuildPrompt(CompletionRequest request)
    {
        var sb = new System.Text.StringBuilder();
        foreach (var m in request.Messages)
        {
            // Extract text content (ignore image parts)
            string text;
            if (m.ContentParts != null)
                text = string.Concat(m.ContentParts.Where(p => p.Type == "text").Select(p => p.Text ?? ""));
            else
                text = m.Content;

            if (m.Role == "system")
                sb.AppendLine(text);
            else
                sb.AppendLine($"[{m.Role}]: {text}");
        }
        return sb.ToString();
    }

    private TcRequest BuildPayload(CompletionRequest request, bool stream) => new()
    {
        Model = request.Model,
        Prompt = BuildPrompt(request),
        MaxTokens = request.MaxTokens,
        Temperature = request.Temperature,
        Stream = stream ? true : null
    };
}

// ── Internal AOT models ────────────────────────────────────────────────────────

internal sealed class TcRequest
{
    [JsonPropertyName("model")] public required string Model { get; init; }
    [JsonPropertyName("prompt")] public required string Prompt { get; init; }
    [JsonPropertyName("max_tokens")] public int MaxTokens { get; init; }
    [JsonPropertyName("temperature")] public double Temperature { get; init; }
    [JsonPropertyName("stream")] public bool? Stream { get; init; }
}

internal sealed class TcResponse
{
    [JsonPropertyName("choices")] public TcChoice[]? Choices { get; init; }
    [JsonPropertyName("usage")] public TcUsage? Usage { get; init; }
}

internal sealed class TcChoice
{
    [JsonPropertyName("text")] public string? Text { get; init; }
    [JsonPropertyName("finish_reason")] public string? FinishReason { get; init; }
}

internal sealed class TcStreamResponse
{
    [JsonPropertyName("choices")] public TcChoice[]? Choices { get; init; }
}

internal sealed class TcUsage
{
    [JsonPropertyName("prompt_tokens")] public int PromptTokens { get; init; }
    [JsonPropertyName("completion_tokens")] public int CompletionTokens { get; init; }
}

[JsonSourceGenerationOptions(DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull)]
[JsonSerializable(typeof(TcRequest))]
[JsonSerializable(typeof(TcResponse))]
[JsonSerializable(typeof(TcStreamResponse))]
internal partial class TextCompletionJsonContext : JsonSerializerContext { }
