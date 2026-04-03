namespace AgentSharp.Core.Interfaces;

/// <summary>
/// Interface for image generation backends (Stable Diffusion WebUI, ComfyUI, etc.)
/// </summary>
public interface IImageGenerationProvider
{
    string Name { get; }

    /// <summary>Generate an image and return raw bytes (JPEG/PNG).</summary>
    Task<ImageGenerationResult> GenerateImageAsync(ImageGenerationRequest request, CancellationToken cancellationToken = default);

    Task<bool> IsAvailableAsync();
}

public sealed class ImageGenerationRequest
{
    public required string Prompt { get; init; }
    public string NegativePrompt { get; init; } = "";
    public int Width { get; init; } = 512;
    public int Height { get; init; } = 512;
    public int Steps { get; init; } = 20;
    public double CfgScale { get; init; } = 7.0;
    public long Seed { get; init; } = -1;
}

public sealed class ImageGenerationResult
{
    public required byte[] ImageData { get; init; }
    public required string MimeType { get; init; }  // "image/jpeg" or "image/png"
    public string? Prompt { get; init; }
    public long Seed { get; init; }
}
