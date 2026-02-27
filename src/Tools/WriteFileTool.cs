using System.Text.Json;
using System.Text.Json.Nodes;
using AgentSharp.Core.Interfaces;

namespace AgentSharp.Core.Tools;

/// <summary>
/// A tool that writes content to a file in GMod
/// </summary>
public sealed class WriteFileTool : ITool
{
    private readonly string _workspacePath;

    public WriteFileTool(string workspacePath)
    {
        _workspacePath = workspacePath;
    }

    public string Name => "write_file";
    public string Description => "Write or overwrite a file. Provide the relative path from the root of your virtual addons folder (e.g. `my_cool_addon/lua/weapons/weapon_cool.lua`).";

    public object Parameters => JsonNode.Parse("""
    {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Relative path to write to (e.g. my_cool_addon/lua/weapons/weapon_cool.lua)"
            },
            "content": {
                "type": "string",
                "description": "The complete Lua code to write"
            }
        },
        "required": [ "path", "content" ]
    }
    """)!;

    public Task<ToolResult> ExecuteAsync(JsonDocument arguments, CancellationToken cancellationToken = default)
    {
        try
        {
            var root = arguments.RootElement;
            var path = root.GetProperty("path").GetString() ?? "";
            var content = root.GetProperty("content").GetString() ?? "";

            // Modify the path to automatically inject the `~` prefix for the root folder
            var parts = path.Split('/', '\\');
            if (parts.Length > 0 && !parts[0].StartsWith("~"))
            {
                parts[0] = "~" + parts[0];
                path = string.Join(Path.DirectorySeparatorChar, parts);
            }

            // Security: Prevent directory traversal
            var fullPath = Path.GetFullPath(Path.Combine(_workspacePath, path));
            if (!fullPath.StartsWith(_workspacePath))
                return Task.FromResult(ToolResult.Fail("Access denied: path escapes workspace."));

            Directory.CreateDirectory(Path.GetDirectoryName(fullPath)!);
            File.WriteAllText(fullPath, content);

            return Task.FromResult(ToolResult.Ok($"Successfully wrote {content.Length} chars to {path}"));
        }
        catch (Exception ex)
        {
            return Task.FromResult(ToolResult.Fail($"Failed to write file: {ex.Message}"));
        }
    }
}