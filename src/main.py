import os
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox, filedialog
import threading

import torch
from PIL import Image, ImageSequence
import numpy as np
from fractions import Fraction
import pathlib

from editor import load_weight, aliasing, upscale, save_png, save_mov

from constants.app import PATH_ICON
from constants.save import Extensions
from constants.upscale import ScaleMethod, DIC_METHOD_TO_SIZE


def execute(files: list[pathlib.Path], scales: list[str], extensions: list[str]):
    file_count = len(files)
    try:
        load_weight()
        for ix, file in enumerate(files):
            dir_save = file.parent / f"{file.stem}"
            dir_save.mkdir(exist_ok=True)
            frames = [
                frame.convert("RGBA")
                for frame in ImageSequence.Iterator(Image.open(file))
            ]
            durations = [frame.info.get("duration", 100) for frame in frames]
            durations, freq = np.unique(durations, return_counts=True)
            duration = durations[np.argmax(freq)]
            rate = Fraction(1000, duration)
            frames_aliased = [aliasing(frame) for frame in frames]
            for scale in scales:
                frames_upscaled = [upscale(frame, scale) for frame in frames_aliased]
                size = DIC_METHOD_TO_SIZE.get(scale)
                for extension in extensions:
                    if extension == Extensions.SEQUENTIAL_PNG:
                        save_png(dir_save / f"{size}", frames_upscaled)
                    elif extension == Extensions.MOV:
                        path_output = dir_save / f"{file.stem}_{size}.mov"
                        save_mov(path_output, frames_upscaled, rate)

            root.after(
                0,
                lambda ix=ix: label_progress.config(
                    text=f"{ix+1}/{file_count} 処理中..."
                ),
            )
            root.after(0, lambda ix=ix: progress.config(value=ix))

    except Exception as e:
        messagebox.showerror("未確認のエラー", f"エラーが発生しました...\n{e}")
        return

    root.after(0, lambda _=ix: progress.config(value=file_count))
    messagebox.showinfo("処理完了", "画像の処理が完了しました！")
    os.startfile(files[0].parent)
    frame_progress.pack_forget()
    btn_execute.pack(pady=20)


def on_select():
    scales = [ScaleMethod.X1, ScaleMethod.X2, ScaleMethod.X4]
    extensions = [Extensions.MOV, Extensions.SEQUENTIAL_PNG]
    scales = [scale for var, scale in zip(vars_scale, scales) if var.get()]
    extensions = [
        extension for var, extension in zip(vars_extension, extensions) if var.get()
    ]

    if not (any(scales) and any(extensions)):
        messagebox.showerror(
            title="未選択エラー",
            message="アップスケール、出力形式にはそれぞれ最低1つチェックを入れてください！",
        )
        return

    types = [("画像ファイル", "*.gif;*.png")]
    files = filedialog.askopenfilenames(
        filetypes=types, title="画像ファイル選択(複数選択可)"
    )

    files = [pathlib.Path(file) for file in files]
    if not files:
        return

    label_progress.config(text=f"1/{len(files)} 処理中...")
    progress["maximum"] = len(files)
    progress.config(value=0)
    btn_execute.pack_forget()
    frame_progress.pack(padx=50, pady=15, fill="both")

    thread = threading.Thread(
        target=execute, args=(files, scales, extensions), daemon=True
    )
    thread.start()


root = tk.Tk()
root.iconbitmap(PATH_ICON)
w_window = 250
h_window = 220
w_screen = root.winfo_screenwidth()
h_screen = root.winfo_screenheight()
x = (w_screen // 2) - (w_window // 2)
y = (h_screen // 2) - (h_window // 2) - 20
root.geometry(f"{w_window}x{h_window}+{x}+{y}")

root.title("MaterialEnhancer")
var_x1 = tk.IntVar()
var_x2 = tk.IntVar()
var_x4 = tk.IntVar()
var_png = tk.IntVar()
var_mov = tk.IntVar()

vars_scale = [var_x1, var_x2, var_x4]
vars_extension = [var_mov, var_png]

available = "有効" if torch.cuda.is_available() else "無効"
label_gpu = tk.Label(root, text=f"GPUサポート：{available}")

frame_input = tk.Frame(root)

frame_upscale = tk.LabelFrame(frame_input, text="アップスケール")
check_x1 = tk.Checkbutton(frame_upscale, text="x1", variable=var_x1)
check_x2 = tk.Checkbutton(frame_upscale, text="x2", variable=var_x2)
check_x4 = tk.Checkbutton(
    frame_upscale,
    text="x4",
    variable=var_x4,
    state="normal" if torch.cuda.is_available() else "disabled",
)
check_x1.pack()
check_x2.pack()
check_x4.pack()

frame_extension = tk.LabelFrame(frame_input, text="出力形式")
check_mov = tk.Checkbutton(frame_extension, text="透過mov", variable=var_mov)
check_png = tk.Checkbutton(frame_extension, text="連番png", variable=var_png)
check_mov.pack()
check_png.pack()

btn_execute = tk.Button(root, text="画像ファイル選択", command=on_select)

frame_progress = tk.Frame(root)
label_progress = tk.Label(frame_progress, text="処理中...")
progress = ttk.Progressbar(frame_progress, mode="determinate")
label_progress.pack()
progress.pack()


label_gpu.pack(pady=15)
frame_input.pack()
frame_upscale.pack(side=tk.LEFT, padx=5, fill="both")
frame_extension.pack(side=tk.LEFT, padx=5, fill="both")
btn_execute.pack(pady=20)

var_x1.set(1)
var_mov.set(1)

if __name__ == "__main__":
    root.mainloop()
