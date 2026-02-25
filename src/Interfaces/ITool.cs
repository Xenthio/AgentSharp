using System.Text.Json;

namespace AgentSharp.Core.Interfaces;

/// <summary>
/// A tool that the AI can call to perform actions
/// </summary>
public interface ITool
{
    string Name { get; }
    string Description { get; }
    object Parameters { get; }

    Task<ToolResult> ExecuteAsync(JsonDocument arguments, CancellationToken cancellationToken = default);

    ToolDefinition GetDefinition() => new()
    {
        Function = new FunctionDefinition
        {
            Name = Name,
            Description = Description,
            Parameters = Parameters
        }
    };
}

public sealed class ToolResult
{
    public required bool Success { get; init; }
    public required string Content { get; init; }
    public string? Error { get; init; }

    public static ToolResult Ok(string content) => new() { Success = true, Content = content };
    public static ToolResult Fail(string error, string? content = null) => new() { Success = false, Error = error, Content = content ?? string.Empty };
}

public sealed class ToolDefinition
{
    [System.Text.Json.Serialization.JsonPropertyName("type")]
    public string Type { get; } = "function";

    [System.Text.Json.Serialization.JsonPropertyName("function")]
    public required FunctionDefinition Function { get; init; }
}

public sealed class FunctionDefinition
{
    [System.Text.Json.Serialization.JsonPropertyName("name")]
    public required string Name { get; init; }

    [System.Text.Json.Serialization.JsonPropertyName("description")]
    public required string Description { get; init; }

    [System.Text.Json.Serialization.JsonPropertyName("parameters")]
    public required object Parameters { get; init; }
}