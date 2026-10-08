"use strict";
/**
 * Module 5 (part 2) -- VS Code Extension
 *
 * Listens for Python file saves, sends the file's content to the local
 * PersonaLint Flask server (Modules 2 + 4), and converts the response
 * into VS Code's native Diagnostic objects -- the same mechanism real
 * linters like ESLint use, so flags appear as yellow squiggly underlines
 * and in the Problems panel.
 *
 * Nothing leaves the developer's machine: the server only listens on
 * localhost.
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.activate = activate;
exports.deactivate = deactivate;
const vscode = require("vscode");
const http = require("http");
let diagnosticCollection;
function activate(context) {
    diagnosticCollection = vscode.languages.createDiagnosticCollection("personalint");
    context.subscriptions.push(diagnosticCollection);
    context.subscriptions.push(vscode.workspace.onDidSaveTextDocument((document) => {
        if (document.languageId === "python") {
            analyzeDocument(document);
        }
    }));
    // Also analyze the currently open Python file on activation, so the
    // demo shows something immediately without requiring an edit+save first.
    if (vscode.window.activeTextEditor?.document.languageId === "python") {
        analyzeDocument(vscode.window.activeTextEditor.document);
    }
}
function analyzeDocument(document) {
    const config = vscode.workspace.getConfiguration("personalint");
    const serverUrl = config.get("serverUrl", "http://localhost:5050");
    const developerId = config.get("developerId", "");
    const body = JSON.stringify({
        code: document.getText(),
        developer_id: developerId || null,
        file_path: document.fileName,
    });
    postJson(`${serverUrl}/analyze`, body)
        .then((response) => {
        const diagnostics = [];
        for (const func of response.functions) {
            if (!func.flagged)
                continue;
            // lineno from Python's AST is 1-indexed; VS Code Positions are 0-indexed.
            const line = Math.max(0, func.lineno - 1);
            const lineText = document.lineAt(Math.min(line, document.lineCount - 1)).text;
            const range = new vscode.Range(new vscode.Position(line, 0), new vscode.Position(line, lineText.length));
            const message = `PersonaLint: '${func.function_name}' looks unusual for ${response.used_personalized_baseline ? "your" : "the population"} coding style:\n` + func.reasons.map((r) => `  - ${r}`).join("\n");
            const diagnostic = new vscode.Diagnostic(range, message, vscode.DiagnosticSeverity.Warning);
            diagnostic.source = "PersonaLint";
            diagnostics.push(diagnostic);
        }
        diagnosticCollection.set(document.uri, diagnostics);
    })
        .catch((err) => {
        console.error("PersonaLint: failed to reach local server.", err);
        vscode.window.setStatusBarMessage("PersonaLint: couldn't reach local server (is flask_api.py running?)", 4000);
    });
}
function postJson(url, body) {
    return new Promise((resolve, reject) => {
        const urlObj = new URL(url);
        const req = http.request({
            hostname: urlObj.hostname,
            port: urlObj.port,
            path: urlObj.pathname,
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Content-Length": Buffer.byteLength(body),
            },
        }, (res) => {
            let data = "";
            res.on("data", (chunk) => (data += chunk));
            res.on("end", () => {
                try {
                    resolve(JSON.parse(data));
                }
                catch (e) {
                    reject(e);
                }
            });
        });
        req.on("error", reject);
        req.write(body);
        req.end();
    });
}
function deactivate() {
    diagnosticCollection?.dispose();
}
//# sourceMappingURL=extension.js.map