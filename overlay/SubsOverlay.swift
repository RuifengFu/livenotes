// subs-overlay：刘海下方的"灵动岛"式实时字幕条。
//
// 读取 subs --live 写的 <标题>.live.json：显示最近两条中文字幕，下方是正在说的英文（实时滚动）。
// 也兼容直接读正在增量写入的 .zh-en.srt（只显示最新一条）。
// 窗口点击穿透、置顶、全屏时也显示；一段时间没有新字幕就收成小胶囊。
//
// 用法：subs-overlay [会话目录或 .zh-en.srt 路径]
//   不给参数时，自动找 ~/Documents/LiveNotes 和当前目录下最新的 *.live.json / *.zh-en.srt
// 结束：subs-overlay 会在字幕文件 2 分钟没有更新且 subs 已退出时自动关闭，或 pkill subs-overlay

import AppKit

struct Cue: Equatable {
    var zh: String
    var en: String
    var prevZh: String = ""
}

/// 解析 live.json：最近两条中文 + 最新英文（有正在说的内容就显示它）。
func liveCue(_ data: Data) -> Cue? {
    guard let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
    let finals = (obj["finals"] as? [[String: Any]]) ?? []
    let partial = (obj["partial"] as? String) ?? ""
    let last = finals.last
    let prev = finals.count >= 2 ? finals[finals.count - 2] : nil
    let en = partial.isEmpty ? ((last?["en"] as? String) ?? "") : partial
    if last == nil && partial.isEmpty { return nil }
    return Cue(zh: (last?["zh"] as? String) ?? "", en: en, prevZh: (prev?["zh"] as? String) ?? "")
}

/// 解析 SRT 最后一条完整字幕。
func lastCue(_ text: String) -> Cue? {
    let blocks = text.components(separatedBy: "\n\n").filter { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
    guard let last = blocks.last else { return nil }
    let lines = last.components(separatedBy: "\n")
    guard lines.count >= 3, lines[1].contains("-->") else { return nil }
    let body = Array(lines[2...]).filter { !$0.isEmpty }
    if body.count >= 2 { return Cue(zh: body[0], en: body[1...].joined(separator: " ")) }
    return Cue(zh: "", en: body.first ?? "")
}

func newestSrt(in dirs: [URL]) -> URL? {
    let fm = FileManager.default
    var best: (URL, Date)?
    for dir in dirs {
        guard let e = fm.enumerator(at: dir, includingPropertiesForKeys: [.contentModificationDateKey],
                                    options: [.skipsHiddenFiles]) else { continue }
        for case let url as URL in e where url.lastPathComponent.hasSuffix(".live.json") || url.lastPathComponent.hasSuffix(".zh-en.srt") {
            let d = (try? url.resourceValues(forKeys: [.contentModificationDateKey]))?.contentModificationDate ?? .distantPast
            if best == nil || d > best!.1 { best = (url, d) }
        }
    }
    return best?.0
}

func resolveSource() -> URL? {
    let args = CommandLine.arguments.dropFirst()
    if let p = args.first {
        let url = URL(fileURLWithPath: (p as NSString).expandingTildeInPath)
        var isDir: ObjCBool = false
        if FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir), isDir.boolValue {
            return newestSrt(in: [url])
        }
        return url
    }
    let home = FileManager.default.homeDirectoryForCurrentUser
    return newestSrt(in: [home.appendingPathComponent("Documents/LiveNotes"),
                          URL(fileURLWithPath: FileManager.default.currentDirectoryPath)])
}

final class Island: NSObject {
    let panel: NSPanel
    let box = NSView()
    let prev = NSTextField(labelWithString: "")
    let zh = NSTextField(labelWithString: "")
    let en = NSTextField(labelWithString: "")
    let dot = NSView()
    var source: URL?
    var current: Cue?
    var lastChange = Date()
    var lastFileSize: UInt64 = 0
    var expanded = false

    let maxWidth: CGFloat
    let topInset: CGFloat

    override init() {
        let screen = NSScreen.main ?? NSScreen.screens[0]
        // 刘海机型：safeAreaInsets.top > 0，字幕条贴在刘海正下方
        topInset = max(screen.safeAreaInsets.top, NSStatusBar.system.thickness)
        maxWidth = min(screen.frame.width * 0.62, 980)
        panel = NSPanel(contentRect: .zero, styleMask: [.borderless, .nonactivatingPanel],
                        backing: .buffered, defer: false)
        super.init()
        panel.isOpaque = false
        panel.backgroundColor = .clear
        panel.hasShadow = true
        panel.level = .screenSaver
        panel.ignoresMouseEvents = true
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .stationary, .ignoresCycle]

        box.wantsLayer = true
        box.layer?.backgroundColor = NSColor.black.withAlphaComponent(0.88).cgColor
        box.layer?.cornerRadius = 22
        box.layer?.cornerCurve = .continuous
        panel.contentView = box

        for (label, size, color, weight) in [(prev, 14.0, NSColor(white: 0.6, alpha: 1), NSFont.Weight.regular),
                                             (zh, 19.0, NSColor.white, NSFont.Weight.semibold),
                                             (en, 13.0, NSColor(white: 0.72, alpha: 1), .regular)] {
            label.font = .systemFont(ofSize: size, weight: weight)
            label.textColor = color
            label.alignment = .center
            label.lineBreakMode = .byWordWrapping
            label.maximumNumberOfLines = 2
            label.cell?.wraps = true
            box.addSubview(label)
        }
        dot.wantsLayer = true
        dot.layer?.backgroundColor = NSColor.systemRed.cgColor
        dot.layer?.cornerRadius = 4
        box.addSubview(dot)

        source = resolveSource()
        layout(animated: false)
        panel.orderFrontRegardless()
        Timer.scheduledTimer(withTimeInterval: 0.25, repeats: true) { [weak self] _ in self?.tick() }
    }

    func tick() {
        if source == nil { source = resolveSource() }
        guard let src = source else { return }
        let size = (try? FileManager.default.attributesOfItem(atPath: src.path)[.size] as? UInt64) ?? 0
        if size != lastFileSize {
            lastFileSize = size
            let cue: Cue? = src.pathExtension == "json"
                ? (try? Data(contentsOf: src)).flatMap(liveCue)
                : (try? String(contentsOf: src, encoding: .utf8)).flatMap(lastCue)
            if let cue, cue != current {
                current = cue
                lastChange = Date()
                prev.stringValue = cue.prevZh
                zh.stringValue = cue.zh
                en.stringValue = cue.en
                layout(animated: true)
            }
        }
        let idle = Date().timeIntervalSince(lastChange)
        // 8 秒没有新字幕：收起成小胶囊（红点表示仍在录）
        if expanded && idle > 8 { layout(animated: true) }
        // 字幕文件很久没更新、subs 也不在跑了：自动退出
        if idle > 120 && !subsRunning() { NSApp.terminate(nil) }
    }

    func subsRunning() -> Bool {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/usr/bin/pgrep")
        p.arguments = ["-x", "subs"]
        p.standardOutput = FileHandle.nullDevice
        try? p.run()
        p.waitUntilExit()
        return p.terminationStatus == 0
    }

    func layout(animated: Bool) {
        let screen = NSScreen.main ?? NSScreen.screens[0]
        let idle = Date().timeIntervalSince(lastChange)
        let show = current != nil && idle <= 8
        expanded = show
        zh.isHidden = !show
        en.isHidden = !show
        prev.isHidden = !show || prev.stringValue.isEmpty
        let pad: CGFloat = 22
        var w: CGFloat = 120, h: CGFloat = 26
        if show {
            let inner = maxWidth - pad * 2
            let zs = zh.cell!.cellSize(forBounds: NSRect(x: 0, y: 0, width: inner, height: 200))
            let es = en.cell!.cellSize(forBounds: NSRect(x: 0, y: 0, width: inner, height: 200))
            let ps = prev.cell!.cellSize(forBounds: NSRect(x: 0, y: 0, width: inner, height: 200))
            let prev_h = prev.stringValue.isEmpty ? 0 : ps.height + 2
            w = min(maxWidth, max(zs.width, es.width, prev_h > 0 ? ps.width : 0) + pad * 2 + 16)
            let zh_h = zh.stringValue.isEmpty ? 0 : zs.height
            h = prev_h + zh_h + es.height + 22
            en.frame = NSRect(x: pad, y: 10, width: w - pad * 2, height: es.height)
            zh.frame = NSRect(x: pad, y: es.height + 12, width: w - pad * 2, height: zh_h)
            prev.frame = NSRect(x: pad, y: es.height + 12 + zh_h, width: w - pad * 2, height: prev_h)
        }
        dot.frame = NSRect(x: show ? 12 : 14, y: h - (show ? 16 : 17), width: 8, height: 8)
        let frame = NSRect(x: screen.frame.midX - w / 2,
                           y: screen.frame.maxY - topInset - h - 6, width: w, height: h)
        box.layer?.cornerRadius = min(22, h / 2)
        if animated {
            NSAnimationContext.runAnimationGroup { ctx in
                ctx.duration = 0.22
                ctx.allowsImplicitAnimation = true
                panel.animator().setFrame(frame, display: true)
            }
        } else {
            panel.setFrame(frame, display: true)
        }
    }
}

let app = NSApplication.shared
app.setActivationPolicy(.accessory)
let island = Island()
signal(SIGINT) { _ in exit(0) }
app.run()
