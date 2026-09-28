// Lightbox.tsx(大图查看):检索结果与图库共用的图片详情浮层。
// Esc / 点击遮罩关闭;传入 onSearchByImage 时展示"以此图检索"动作(图库页跳检索页用)

import { useEffect } from "react";
import { ArrowsOut, MagnifyingGlass, Trash, X } from "@phosphor-icons/react";

export interface LightboxImage {
  imageUrl: string;
  title?: string;
  description?: string | null;
  relevanceScore?: number;
  metadata?: Record<string, unknown> | null;
}

interface LightboxProps {
  image: LightboxImage;
  onClose: () => void;
  onSearchByImage?: (imageUrl: string) => void;
  /** onDelete 存在时展示删除按钮(图库页管理用;确认逻辑由调用方负责) */
  onDelete?: () => void;
}

// formatBytes(字节数可读化):图库页与详情共用
export function formatBytes(bytes: number): string {
  if (bytes >= 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${bytes} B`;
}

export default function Lightbox({ image, onClose, onSearchByImage, onDelete }: LightboxProps) {
  // Esc 关闭:浮层类组件的键盘逃生口
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const fileName = String(image.metadata?.file_name ?? image.title ?? "");

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-zinc-950/70 p-4 backdrop-blur-sm"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-label={image.title ?? "图片详情"}
    >
      <div
        className="flex max-h-[88vh] w-full max-w-4xl flex-col overflow-hidden rounded-xl bg-white shadow-2xl md:flex-row"
        onClick={(e) => e.stopPropagation()}
      >
        {/* 图片区:object-contain 保证长图/宽图都完整可见 */}
        <div className="flex min-h-[240px] flex-1 items-center justify-center bg-zinc-950 p-3">
          <img
            src={image.imageUrl}
            alt={image.title ?? "检索图片"}
            className="max-h-[80vh] max-w-full object-contain"
          />
        </div>

        {/* 信息区 */}
        <div className="flex w-full shrink-0 flex-col gap-4 overflow-y-auto p-5 md:w-72">
          <div className="flex items-start justify-between gap-2">
            <h3 className="text-sm leading-snug font-semibold break-all">
              {image.title || "未命名图片"}
            </h3>
            <button
              onClick={onClose}
              className="-m-1 rounded-md p-1 text-zinc-400 transition-colors hover:bg-zinc-100 hover:text-zinc-900"
              aria-label="关闭"
            >
              <X size={18} />
            </button>
          </div>

          {image.relevanceScore !== undefined && (
            <div>
              <p className="text-xs text-zinc-500">相似度</p>
              <p className="font-mono text-2xl font-semibold text-emerald-700">
                {image.relevanceScore.toFixed(3)}
              </p>
            </div>
          )}

          <div>
            <p className="mb-1 text-xs text-zinc-500">图片描述</p>
            <p className="text-sm leading-relaxed text-zinc-700">
              {image.description?.trim() || "暂无描述。真实检索模式下会展示模型生成的图片说明。"}
            </p>
          </div>

          {fileName && (
            <div>
              <p className="mb-1 text-xs text-zinc-500">文件</p>
              <p className="font-mono text-xs break-all text-zinc-500">{fileName}</p>
            </div>
          )}

          <div className="mt-auto flex flex-col gap-2 pt-2">
            {onSearchByImage && (
              <button
                onClick={() => onSearchByImage(image.imageUrl)}
                className="flex items-center justify-center gap-2 rounded-lg bg-zinc-900 px-4 py-2.5 text-sm font-medium text-white transition-all hover:bg-zinc-700 active:scale-[0.98]"
              >
                <MagnifyingGlass size={16} />
                以此图检索
              </button>
            )}
            <a
              href={image.imageUrl}
              target="_blank"
              rel="noreferrer"
              className="flex items-center justify-center gap-2 rounded-lg border border-zinc-200 px-4 py-2.5 text-sm text-zinc-700 transition-colors hover:bg-zinc-50"
            >
              <ArrowsOut size={16} />
              原图打开
            </a>
            {onDelete && (
              <button
                onClick={onDelete}
                className="flex items-center justify-center gap-2 rounded-lg border border-red-200 px-4 py-2.5 text-sm text-red-700 transition-colors hover:bg-red-50"
              >
                <Trash size={16} />
                删除图片
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
