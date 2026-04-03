using System.Net.Http.Json;
using System.Text.Json;
using System.Text.Json.Serialization;
using AgentSharp.Core.Interfaces;

namespace AgentSharp.Core.Backends;

/// <summary>
/// Stable Diffusion WebUI (Automatic1111) image generation backend.
/// Connects to POST /sdapi/v1/txt2img.
/// </summary>
public sealed class StableDiffusionBackend : IImageGenerationProvider
{
    private readonly HttpClient _client;

    public string Name => "stable-diffusion";

    public StableDiffusionBackend(string? baseUrl = null)
    {
        var url = (baseUrl ?? "http://127.0.0.1:7860").TrimEnd('/') + "/";
        _client = new HttpClient
        {
            BaseAddress = new Uri(url),
            Timeout = TimeSpan.FromMinutes(10)
        };
    }

    public async Task<ImageGenerationResult> GenerateImageAsync(
        ImageGenerationRequest request,
        CancellationToken cancellationToken = default)
    {
        var payload = new SdTxt2ImgRequest
        {
            Prompt = request.Prompt,
            NegativePrompt = request.NegativePrompt,
            Width = request.Width,
            Height = request.Height,
            Steps = request.Steps,
            CfgScale = request.CfgScale,
            Seed = request.Seed
        };

        var resp = await _client.PostAsJsonAsync("sdapi/v1/txt2img", payload,
            SdJsonContext.Default.SdTxt2ImgRequest, cancellationToken);

        var raw = await resp.Content.ReadAsStringAsync(cancellationToken);
        if (!resp.IsSuccessStatusCode)
            throw new Exception($"[{Name}] API Error ({resp.StatusCode}): {raw}");

        var result = JsonSerializer.Deserialize(raw, SdJsonContext.Default.SdTxt2ImgResponse);
        if (result?.Images == null || result.Images.Length == 0)
            throw new Exception($"[{Name}] No images returned.");

        var imageData = Convert.FromBase64String(result.Images[0]);
        return new ImageGenerationResult
        {
            ImageData = imageData,
            MimeType = "image/png",
            Prompt = request.Prompt,
            Seed = request.Seed
        };
    }

    public async Task<bool> IsAvailableAsync()
    {
        try
        {
            var resp = await _client.GetAsync("sdapi/v1/samplers");
            return resp.IsSuccessStatusCode;
        }
        catch { return false; }
    }
}

// ── Internal AOT models ────────────────────────────────────────────────────────

internal sealed class SdTxt2ImgRequest
{
    [JsonPropertyName("prompt")] public required string Prompt { get; init; }
    [JsonPropertyName("negative_prompt")] public string NegativePrompt { get; init; } = "";
    [JsonPropertyName("width")] public int Width { get; init; } = 512;
    [JsonPropertyName("height")] public int Height { get; init; } = 512;
    [JsonPropertyName("steps")] public int Steps { get; init; } = 20;
    [JsonPropertyName("cfg_scale")] public double CfgScale { get; init; } = 7.0;
    [JsonPropertyName("seed")] public long Seed { get; init; } = -1;
}

internal sealed class SdTxt2ImgResponse
{
    [JsonPropertyName("images")] public string[]? Images { get; init; }
}

[JsonSourceGenerationOptions(DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull)]
[JsonSerializable(typeof(SdTxt2ImgRequest))]
[JsonSerializable(typeof(SdTxt2ImgResponse))]
internal partial class SdJsonContext : JsonSerializerContext { }
