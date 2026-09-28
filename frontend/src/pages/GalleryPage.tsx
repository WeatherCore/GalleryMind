// GalleryPage.tsx(图片库):浏览与管理后端图片库。
// 数据:GET /api/images(含 caption);管理:POST upload / DELETE / POST reindex(热重建)
// 上传入库后真实模式需重建索引才能被检索,mock 模式下重建返回 skipped 提示

import { useCallback, useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  ArrowClockwise,
  Images,
  UploadSimple,
  WarningCircle,
} from "@phosphor-icons/react";
import { deleteImage, fetchImages, reindexImages, uploadImage, urlToBase64 } from "../api";
import type { GalleryImage } from "../types";
import Lightbox, { formatBytes } from "../components/Lightbox";

export default function GalleryPage() {
  const [images, setImages] = useState<GalleryImage[] | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState<GalleryImage | null>(null);

  // 上传与重建的进行中状态及结果提示(成功提示用中性色,常驻到下次操作)
  const [uploading, setUploading] = useState(false);
  const [reindexing, setReindexing] = useState(false);
  const [notice, setNotice] = useState<{ kind: "ok" | "info" | "error"; text: string } | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const navigate = useNavigate();

  const load = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const resp = await fetchImages();
      setImages(resp.images);
    } catch (e) {
      setError(e instanceof Error ? e.message : "加载图片库失败");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  // handleUpload:支持一次选多张,逐张上传,结束后刷新列表并提示是否需要重建索引
  const handleUpload = async (files: FileList | null) => {
    if (!files?.length) return;
    setUploading(true);
    setNotice(null);
    let ok = 0;
    let fail = 0;
    for (const file of Array.from(files)) {
      try {
        await uploadImage(file);
        ok += 1;
      } catch {
        fail += 1;
      }
    }
    setUploading(false);
    setNotice({
      kind: fail ? "error" : "ok",
      text: fail
        ? `${ok} 张上传成功,${fail} 张失败`
        : `已上传 ${ok} 张图片。真实检索模式下需点击"重建索引"才能被搜到`,
    });
    await load();
  };

  // handleReindex:真实模式同步等待重建(可能数十秒),mock 模式后端直接返回 skipped
  const handleReindex = async () => {
    setReindexing(true);
    setNotice(null);
    try {
      const resp = await reindexImages();
      setNotice({ kind: resp.status === "ok" ? "ok" : "info", text: resp.message ?? `状态: ${resp.status}` });
    } catch (e) {
      setNotice({ kind: "error", text: e instanceof Error ? e.message : "重建请求失败" });
    } finally {
      setReindexing(false);
    }
  };

  // handleDelete:破坏性操作,弹确认;删除后顺带关掉浮层
  const handleDelete = async (filename: string) => {
    if (!window.confirm(`确定删除图片「${filename}」吗?该操作不可恢复。`)) return;
    try {
      await deleteImage(filename);
      setSelected(null);
      setNotice({ kind: "ok", text: `已删除 ${filename}` });
      await load();
    } catch (e) {
      setNotice({ kind: "error", text: e instanceof Error ? e.message : "删除失败" });
    }
  };

  // 以图搜图:取图片转 base64,带着跳检索页
  const searchWithImage = async (imageUrl: string) => {
    setSelected(null);
    try {
      const b64 = await urlToBase64(imageUrl);
      navigate("/", { state: { imageB64: b64 } });
    } catch {
      setError("读取图片失败,无法以此图检索");
    }
  };

  return (
    <div className="mx-auto max-w-6xl px-4 py-8 md:px-8">
      <header className="mb-5 flex items-end justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold tracking-tight">图片库</h1>
          <p className="mt-1 text-sm text-zinc-500">
            {images
              ? `共 ${images.length} 张图片,点击任意图片可查看详情或发起以图搜图`
              : "backend/data/images 下的全部可检索图片"}
          </p>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          {/* 上传:隐藏 input + 按钮触发,支持多选 */}
          <input
            ref={fileInputRef}
            type="file"
            accept="image/png,image/jpeg,image/gif"
            multiple
            className="hidden"
            onChange={(e) => {
              void handleUpload(e.target.files);
              e.target.value = ""; // 允许连续上传同一文件
            }}
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={uploading}
            className="flex items-center gap-1.5 rounded-lg bg-zinc-900 px-3 py-2 text-sm font-medium text-white transition-all hover:bg-zinc-700 active:scale-[0.98] disabled:opacity-50"
          >
            <UploadSimple size={15} />
            {uploading ? "上传中" : "上传图片"}
          </button>
          <button
            onClick={() => void handleReindex()}
            disabled={reindexing}
            title="用当前图片库重建检索索引(无需重启后端)"
            className="flex items-center gap-1.5 rounded-lg border border-zinc-200 bg-white px-3 py-2 text-sm text-zinc-700 transition-colors hover:bg-zinc-50 disabled:opacity-50"
          >
            <ArrowClockwise size={15} className={reindexing ? "animate-spin" : ""} />
            {reindexing ? "重建中" : "重建索引"}
          </button>
          <button
            onClick={() => void load()}
            disabled={loading}
            className="flex items-center gap-1.5 rounded-lg border border-zinc-200 bg-white px-3 py-2 text-sm text-zinc-700 transition-colors hover:bg-zinc-50 disabled:opacity-50"
          >
            刷新
          </button>
        </div>
      </header>

      {/* 操作结果提示 */}
      {notice && (
        <div
          className={`mb-5 rounded-lg border px-4 py-3 text-sm ${
            notice.kind === "ok"
              ? "border-emerald-200 bg-emerald-50 text-emerald-800"
              : notice.kind === "info"
                ? "border-zinc-200 bg-zinc-50 text-zinc-700"
                : "border-red-200 bg-red-50 text-red-800"
          }`}
        >
          {notice.text}
        </div>
      )}

      {error && (
        <div className="mb-6 flex items-start gap-2.5 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800">
          <WarningCircle size={18} className="mt-0.5 shrink-0" />
          <div>
            <p className="font-medium">图片库加载失败</p>
            <p className="mt-0.5 text-red-700">{error}</p>
          </div>
        </div>
      )}

      {loading && !images ? (
        <div className="grid grid-cols-1 gap-x-4 gap-y-6 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          {Array.from({ length: 8 }).map((_, i) => (
            <div key={i}>
              <div className="aspect-[4/3] animate-pulse rounded-lg bg-zinc-200/70" />
              <div className="mt-2 h-4 w-2/3 animate-pulse rounded bg-zinc-200/70" />
            </div>
          ))}
        </div>
      ) : images && images.length > 0 ? (
        <div className="grid grid-cols-1 gap-x-4 gap-y-6 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          {images.map((img, i) => (
            <button
              key={img.filename}
              onClick={() => setSelected(img)}
              className="rise-in group text-left"
              style={{ "--index": Math.min(i, 12) } as React.CSSProperties}
            >
              <div className="aspect-[4/3] overflow-hidden rounded-lg border border-zinc-200 bg-white transition-shadow group-hover:shadow-md">
                <img
                  src={img.url}
                  alt={img.filename}
                  loading="lazy"
                  className="h-full w-full object-cover transition-transform duration-300 group-hover:scale-[1.03]"
                />
              </div>
              <div className="mt-2 flex items-baseline justify-between gap-2">
                <p className="truncate font-mono text-xs text-zinc-600">{img.filename}</p>
                <span className="shrink-0 font-mono text-xs text-zinc-400">
                  {formatBytes(img.sizeBytes)}
                </span>
              </div>
            </button>
          ))}
        </div>
      ) : images ? (
        <div className="mt-16 flex flex-col items-center text-center">
          <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-zinc-100">
            <Images size={26} className="text-zinc-400" />
          </div>
          <p className="mt-4 text-sm font-medium text-zinc-700">图片库为空</p>
          <p className="mt-1 max-w-sm text-sm text-zinc-500">
            点击右上角"上传图片"添加,或把图片放进 backend/data/images 后点击刷新
          </p>
        </div>
      ) : null}

      {selected && (
        <Lightbox
          image={{
            imageUrl: selected.url,
            title: selected.filename,
            description: selected.hasCaption ? selected.caption : null,
            metadata: { file_name: selected.filename },
          }}
          onClose={() => setSelected(null)}
          onSearchByImage={searchWithImage}
          onDelete={() => void handleDelete(selected.filename)}
        />
      )}
    </div>
  );
}
