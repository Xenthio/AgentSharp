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
    public double Temperature { get; init; } = 0.7;
    public int MaxTokens { get; init; } = 4096;
    public List<ToolDefinition>? Tools { get; init; }
}

public sealed class ChatMessage
{
    public required string Role { get; init; }
    public required string Content { get; set; }
    public string? Name { get; init; }
    public List<ToolCall>? ToolCalls { get; init; }
    public string? ToolCallId { get; init; }

    public static ChatMessage User(string text) => new() { Role = "user", Content = text };
    public static ChatMessage System(string text) => new() { Role = "system", Content = text };
    public static ChatMessage Assistant(string text) => new() { Role = "assistant", Content = text };
    public static ChatMessage AssistantTools(List<ToolCall> tools) => new() { Role = "assistant", Content = "", ToolCalls = tools };
    public static ChatMessage ToolResult(string id, string result) => new() { Role = "tool", ToolCallId = id, Content = result };
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