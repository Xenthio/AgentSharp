using System.Text.Json;

namespace AgentSharp.Core.Interfaces;

/// <summary>
/// Universal LLM Provider Interface
/// </summary>
public interface IBackendProvider
{
    string Name { get; }
    
    Task<CompletionResponse> GenerateCompletionAsync(
        CompletionRequest request, 
        CancellationToken cancellationToken = default);
        
    IAsyncEnumerable<CompletionStreamChunk> GenerateCompletionStreamAsync(
        CompletionRequest request,
        CancellationToken cancellationToken = default);
        
    Task<float[]> GenerateEmbeddingAsync(
        string input,
        string model = "google/gemini-embedding-001",
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Count the number of tokens in a string using the backend's tokenizer.
    /// Returns -1 if the backend does not support tokenization.
    /// </summary>
    Task<int> CountTokensAsync(string text, CancellationToken cancellationToken = default);
}

public sealed class CompletionStreamChunk
{
    public string? ContentDelta { get; init; }
    public string? ReasoningDelta { get; init; }
    public List<ToolCall>? ToolCalls { get; init; }
    public TokenUsage? Usage { get; init; }
    public string? FinishReason { get; init; }
}

public sealed class CompletionRequest
{
    public required string Model { get; init; }
    public required List<ChatMessage> Messages { get; init; }
    public double Temperature   { get; init; } = 0.7;
    public int    MaxTokens     { get; init; } = 4096;
    public List<ToolDefinition>? Tools { get; init; }

    // Generation sampling parameters (0/default = use model default)
    public double MinP          { get; init; } = 0.0;
    public int    TopK          { get; init; } = 0;
    public double TopP          { get; init; } = 1.0;
    public double RepeatPenalty { get; init; } = 1.0;
    public int    Seed          { get; init; } = -1;

    // Thinking / extended reasoning
    public bool EnableThinking  { get; init; } = false;
    public int  ThinkingBudget  { get; init; } = 0;
    /// <summary>Token ID → logit bias (-100 to 100). Use to ban or boost specific tokens.</summary>
    public Dictionary<string, float>? LogitBias { get; init; }
}

public sealed class ChatMessage
{
    public required string Role { get; init; }
    public required string Content { get; set; }
    public string? Name { get; init; }
    public List<ToolCall>? ToolCalls { get; init; }
    public string? ToolCallId { get; init; }
    public List<ChatContentPart>? ContentParts { get; init; }

    public static ChatMessage User(string text) => new() { Role = "user", Content = text };
    public static ChatMessage System(string text) => new() { Role = "system", Content = text };
    public static ChatMessage Assistant(string text) => new() { Role = "assistant", Content = text };
    public static ChatMessage AssistantTools(List<ToolCall> tools) => new() { Role = "assistant", Content = "", ToolCalls = tools };
    public static ChatMessage ToolResult(string id, string result) => new() { Role = "tool", ToolCallId = id, Content = result };

    /// <summary>
    /// Creates a user message with multimodal content (text + images).
    /// </summary>
    public static ChatMessage UserWithImages(string text, params ChatContentPart[] parts)
    {
        var allParts = new List<ChatContentPart> { ChatContentPart.FromText(text) };
        allParts.AddRange(parts);
        return new() { Role = "user", Content = text, ContentParts = allParts };
    }
}

/// <summary>
/// A single part of a multimodal message content array.
/// </summary>
public sealed class ChatContentPart
{
    public required string Type { get; init; }  // "text" or "image_url"
    public string? Text { get; init; }
    public ChatImageUrl? ImageUrl { get; init; }

    public static ChatContentPart FromText(string text) => new() { Type = "text", Text = text };
    public static ChatContentPart FromImageUrl(string url, string detail = "auto") =>
        new() { Type = "image_url", ImageUrl = new ChatImageUrl { Url = url, Detail = detail } };
    public static ChatContentPart FromBase64Image(byte[] data, string mimeType = "image/jpeg", string detail = "auto") =>
        FromImageUrl($"data:{mimeType};base64,{Convert.ToBase64String(data)}", detail);
}

public sealed class ChatImageUrl
{
    public required string Url { get; init; }
    public string Detail { get; init; } = "auto";  // "auto" | "low" | "high"
}

public sealed class ToolCall
{
    public required string Id { get; init; }
    public required string Type { get; init; }
    public required ToolCallFunction Function { get; init; }
}

public sealed class ToolCallFunction
{
    public required string Name { get; init; }
    public required string Arguments { get; set; }
}

public sealed class CompletionResponse
{
    public string Content { get; init; } = string.Empty;
    /// <summary>Reasoning/thinking content if the model returned it (e.g. Qwen3, DeepSeek R1 reasoning_content).</summary>
    public string? ReasoningContent { get; init; }
    public List<ToolCall>? ToolCalls { get; init; }
    public required TokenUsage Usage { get; init; }
    public required string FinishReason { get; init; }
}

public sealed class TokenUsage
{
    public int PromptTokens { get; init; }
    public int CompletionTokens { get; init; }
    public int TotalTokens { get; init; }
}