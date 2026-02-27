using System.Text.Json.Serialization;

namespace AgentSharp.Core.Backends;

public sealed class OpenRouterEmbeddingRequest
{
    [JsonPropertyName("model")]
    public required string Model { get; init; }

    [JsonPropertyName("input")]
    public required string Input { get; init; }
}

public sealed class OpenRouterEmbeddingResponse
{
    [JsonPropertyName("data")]
    public EmbeddingData[]? Data { get; init; }
}

public sealed class EmbeddingData
{
    [JsonPropertyName("embedding")]
    public float[]? Embedding { get; init; }
}

[JsonSerializable(typeof(OpenRouterEmbeddingRequest))]
[JsonSerializable(typeof(OpenRouterEmbeddingResponse))]
internal partial class OpenRouterEmbeddingJsonContext : JsonSerializerContext
{
}
