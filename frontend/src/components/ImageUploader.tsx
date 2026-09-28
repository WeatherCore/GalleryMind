// ImageUploader.tsx(图片上传):图搜图与 Agent 传图共用的上传控件。
// 支持三种输入方式:点击选文件 / 拖拽放入 / 全局 Ctrl+V 粘贴截图(演示时最顺手的路径)
// 输出统一为 base64 data URL,直接对接后端 uploadedImage / image 字段

import { useEffect, useRef, useState } from "react";
import { ImageSquare, X } from "@phosphor-icons/react";
import { fileToBase64 } from "../api";

interface ImageUploaderProps {
  value: string | null;
  onChange: (base64: string | null) => void;
  /** compact:Agent 输入框用的小按钮形态(无拖拽大框) */
  compact?: boolean;
}

export default function ImageUploader({ value, onChange, compact = false }: ImageUploaderProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);
  const [error, setError] = useState("");

  const acceptFile = async (file: File | undefined | null) => {
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      setError("只能上传图片文件");
      return;
    }
    setError("");
    onChange(await fileToBase64(file));
  };

  // 全局粘贴监听:有图无图都响应,Ctrl+V 直接贴截图。
  // 注意 cleanup,否则页面切换后监听器泄漏、重复回调
  useEffect(() => {
    const onPaste = (e: ClipboardEvent) => {
      const item = Array.from(e.clipboardData?.items ?? []).find((i) =>
        i.type.startsWith("image/"),
      );
      if (item) {
        const file = item.getAsFile();
        if (file) {
          e.preventDefault();
          void acceptFile(file);
        }
      }
    };
    window.addEventListener("paste", onPaste);
    return () => window.removeEventListener("paste", onPaste);
  }, []);

  const pickFile = () => inputRef.current?.click();

  if (compact) {
    return (
      <>
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(e) => void acceptFile(e.target.files?.[0])}
        />
        <button
          onClick={pickFile}
          className={`rounded-lg p-2 transition-colors ${
            value
              ? "bg-emerald-50 text-emerald-700"
              : "text-zinc-400 hover:bg-zinc-100 hover:text-zinc-700"
          }`}
          title={value ? "已附带图片(点击更换)" : "附带图片"}
          aria-label="附带图片"
        >
          <ImageSquare size={18} />
        </button>
      </>
    );
  }

  if (value) {
    return (
      <div className="flex flex-col gap-1.5">
        <div className="relative inline-block w-fit">
          <img
            src={value}
            alt="待检索图片预览"
            className="h-24 w-24 rounded-lg border border-zinc-200 object-cover"
          />
          <button
            onClick={() => onChange(null)}
            className="absolute -top-1.5 -right-1.5 rounded-full bg-zinc-900 p-0.5 text-white shadow transition-transform hover:scale-110"
            aria-label="移除图片"
          >
            <X size={12} weight="bold" />
          </button>
        </div>
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(e) => void acceptFile(e.target.files?.[0])}
        />
        {error && <p className="text-xs text-red-600">{error}</p>}
      </div>
    );
  }

  return (
    <div
      onClick={pickFile}
      onDragOver={(e) => {
        e.preventDefault();
        setDragOver(true);
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragOver(false);
        void acceptFile(e.dataTransfer.files?.[0]);
      }}
      className={`flex cursor-pointer flex-col items-center justify-center gap-1.5 rounded-lg border border-dashed px-4 py-6 text-center transition-colors ${
        dragOver
          ? "border-emerald-500 bg-emerald-50"
          : "border-zinc-300 hover:border-zinc-400 hover:bg-zinc-50"
      }`}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") pickFile();
      }}
    >
      <ImageSquare size={22} className="text-zinc-400" />
      <p className="text-sm text-zinc-600">
        点击选择、拖入图片,或直接 <kbd className="font-mono text-xs">Ctrl+V</kbd> 粘贴截图
      </p>
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(e) => void acceptFile(e.target.files?.[0])}
      />
      {error && <p className="text-xs text-red-600">{error}</p>}
    </div>
  );
}
