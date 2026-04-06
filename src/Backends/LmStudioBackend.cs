using System.Net.Http.Json;
using System.Runtime.CompilerServices;
using System.Text.Json;
using AgentSharp.Core.Interfaces;

namespace AgentSharp.Core.Backends;

/// <summary>
/// LM Studio backend (OpenAI-compatible API).
/// Auto-detects chat-completions vs text-completions on first use.
/// Default base URL: http://127.0.0.1:1234
/// </summary>
public sealed class LmStudioBackend : IBackendProvider
{
    public string Name => "lmstudio";

    private readonly string _baseUrl;
    private IBackendProvider? _impl;

    private static readonly HttpClient _probeHttp = new() { Timeout = TimeSpan.FromSeconds(5) };

    public LmStudioBackend(string baseUrl = "http://127.0.0.1:1234")
    {
        _baseUrl = baseUrl.TrimEnd('/');
    }

    public async Task<CompletionResponse> GenerateCompletionAsync(
        CompletionRequest request,
        CancellationToken cancellationToken = default)
    {
        await EnsureImplAsync(cancellationToken);
        return await _impl!.GenerateCompletionAsync(request, cancellationToken);
    }

    public async IAsyncEnumerable<CompletionStreamChunk> GenerateCompletionStreamAsync(
        CompletionRequest request,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        await EnsureImplAsync(cancellationToken);
        await foreach (var chunk in _impl!.GenerateCompletionStreamAsync(request, cancellationToken))
            yield return chunk;
    }

    public async Task<float[]> GenerateEmbeddingAsync(
        string input,
        string model = "google/gemini-embedding-001",
        CancellationToken cancellationToken = default)
    {
        await EnsureImplAsync(cancellationToken);
        return await _impl!.GenerateEmbeddingAsync(input, model, cancellationToken);
    }

    public async Task<int> CountTokensAsync(string text, CancellationToken cancellationToken = default)
    {
        await EnsureImplAsync(cancellationToken);
        return await _impl!.CountTokensAsync(text, cancellationToken);
    }

    public async Task<bool> IsAvailableAsync()
    {
        try
        {
            var resp = await _probeHttp.GetAsync($"{_baseUrl}/v1/models");
            return resp.IsSuccessStatusCode;
        }
        catch { return false; }
    }

    private async Task EnsureImplAsync(CancellationToken ct)
    {
        if (_impl != null) return;

        bool chatOk = false;
        try
        {
            using var cts = CancellationTokenSource.CreateLinkedTokenSource(ct);
            cts.CancelAfter(TimeSpan.FromSeconds(5));

            var testPayload = new
            {
                model = string.Empty,
                messages = new[] { new { role = "user", content = "hi" } },
                max_tokens = 1,
            };
            using var content = JsonContent.Create(testPayload);
            var resp = await _probeHttp.PostAsync($"{_baseUrl}/v1/chat/completions", content, cts.Token);
            chatOk = resp.IsSuccessStatusCode;
        }
        catch { }

        if (chatOk)
        {
            Console.WriteLine($"[{Name}] Using chat-completions API.");
            _impl = new OpenAiCompatibleBackend(_baseUrl, apiKey: null);
        }
        else
        {
            Console.WriteLine($"[{Name}] Using text-completions API.");
            _impl = new TextCompletionBackend(_baseUrl);
        }
    }
}
