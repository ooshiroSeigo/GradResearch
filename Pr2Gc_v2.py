import os
import json
import numpy as np
import gcoordinator as gc
import google.generativeai as genai
from google.ai import generativelanguage as gl

# APIキーの設定
import os
from dotenv import load_dotenv

# .envファイルを読み込む
load_dotenv()

# 環境変数からAPIキーを取得
api_key = os.getenv("API_KEY")
genai.configure(api_key=api_key)

def generate_3d_equation(user_request: str) -> dict:
    print("[1/5] Gemini API にリクエストを送信中（シンプルな数式の取得）...")
    
    system_prompt = (
        "You are an expert mathematician. Convert the user's 3D printing request into a single mathematical inequality. "
        "Output strict JSON matching the response schema."
    )
    
    # 完全にシンプルな構造（数式文字列、最大高さ、造形サイズ）だけを指定
    response_schema = gl.Schema(
        type=gl.Type.OBJECT,
        properties={
            "formula": gl.Schema(
                type=gl.Type.OBJECT,
                properties={
                    "expr": gl.Schema(type=gl.Type.STRING, description="物体の内側を表すPythonの条件式（例: x**2 + y**2 + (z-3)**2 <= 3**2）"),
                    "max_z": gl.Schema(type=gl.Type.NUMBER, description="オブジェクト全体の最大の高さ(mm)"),
                    "size_xy": gl.Schema(type=gl.Type.NUMBER, description="オブジェクトの横幅の最大値(mm)。この範囲内をスキャンします（例: 30.0）")
                },
                required=["expr", "max_z", "size_xy"]
            )
        },
        required=["formula"]
    )
    
    model = genai.GenerativeModel(
        model_name='models/gemini-3.5-flash-lite', 
        system_instruction=system_prompt
    )
    
    user_prompt = f"""
    ユーザーの要求: {user_request}

    【数式表現のルール】
    - 変数 `x`, `y`, `z` を用いて、物体の「内側」を判定する不等式の文字列を作ってください。
    - 必ずオブジェクトの最下部が Z = 0 に接地するように、Zの座標をプラス方向にオフセットしてください。
    - 例1（半径3mmの球）: "x**2 + y**2 + (z - 3)**2 <= 3**2" (max_z: 6.0, size_xy: 10.0)
    - 例2（一辺20mmの正方形の筒）: "(np.abs(x) <= 10) & (np.abs(y) <= 10)" (max_z: 50.0, size_xy: 25.0)
    - 反応を安定させるため、生の Python の if 文や改行コード(\\n)は絶対に使わず、1行の数式（不等式）のみにしてください。
    """
    
    response = model.generate_content(
        user_prompt,
        generation_config={
            "response_mime_type": "application/json",
            "response_schema": response_schema,
            "temperature": 0.2
        }
    )
    return json.loads(response.text.strip())

# 事前にベース設定を読み込む
settings_path = "/Users/oosiroseigo/ゼミ/卒研用ディレクトリ/base_settings.json"
gc.load_settings(settings_path)

with open(settings_path, "r", encoding="utf-8") as f:
    base_cfg = json.load(f)
LAYER_HEIGHT = base_cfg.get("Print", {}).get("layer_height", 0.2) 

if __name__ == "__main__":
    print("==================================================")
    user_input = input("作製したい3Dオブジェクトの形状や特徴を入力してください:\n> ")
    print("==================================================")
    
    if not user_input.strip():
        print("❌ 入力が空です。処理を中断します。")
        exit()
        
    try:
        # 1. Geminiから数式データを取得
        data = generate_3d_equation(user_input)
        expr = data["formula"]["expr"]
        max_z = float(data["formula"]["max_z"])
        size_xy = float(data["formula"]["size_xy"])
        
        print(f"[2/5] Geminiが生成した不等式: {expr}")
        print(f"📏 最大高さ: {max_z}mm | スキャン範囲: ±{size_xy}mm")
        print("[3/5] 空間グリッドを生成し、一層ずつ独立したパスとしてスライス中...")
        
        # 【変更】全層一括のリストではなく、1層ごとのgc.Pathオブジェクトを格納するリストを用意
        path_list = []
        
        # 輪郭を走査するための角度展開（360方向）
        angles = np.linspace(0, 2 * np.pi, 360)
        
        # Z軸方向に下から順にスライス処理
        current_z = LAYER_HEIGHT
        while current_z <= max_z:
            z = current_z
            
            # このレイヤーのX, Yの細かな格子（碁盤の目）を作る（解像度0.2mm）
            grid_range = np.arange(-size_xy, size_xy, 0.2)
            mx, my = np.meshgrid(grid_range, grid_range)
            x_flat, y_flat = mx.flatten(), my.flatten()
            z_flat = np.full_like(x_flat, z)
            
            # 1行の数式を一括計算
            eval_env = {"x": x_flat, "y": y_flat, "z": z_flat, "np": np}
            is_inside = eval(expr, {"__builtins__": None}, eval_env)
            
            # 物体の内側（True）だった座標だけを抽出
            inside_x = x_flat[is_inside]
            inside_y = y_flat[is_inside]
            
            if len(inside_x) > 0:
                layer_x, layer_y = [], []
                
                # 360方向に向かって「一番外側にある点」だけをつまみ食いして一筆書きにする
                for theta in angles:
                    dists = inside_x * np.cos(theta) + inside_y * np.sin(theta)
                    if len(dists) > 0:
                        max_idx = np.argmax(dists)
                        layer_x.append(inside_x[max_idx])
                        layer_y.append(inside_y[max_idx])
                
                if layer_x:
                    # 始点を末尾にコピーして綺麗に閉じる
                    layer_x.append(layer_x[0])
                    layer_y.append(layer_y[0])
                    layer_z = [z] * len(layer_x)
                    
                    # 【重要】この1層分だけの座標から、独立した gc.Path オブジェクトを作成
                    layer_path = gc.Path(np.array(layer_x), np.array(layer_y), np.array(layer_z))
                    layer_path.apply_default_settings() # 層ごとに設定を適用
                    
                    # リストに追加
                    path_list.append(layer_path)
            
            current_z += LAYER_HEIGHT
            
        if len(path_list) == 0:
            raise ValueError("形状が検出されませんでした。数式またはスキャン範囲を見直してください。")
            
        print(f"[4/5] gcoordinator パス生成中... (総レイヤー数: {len(path_list)} 層)")
        
        # 【超重要】エラーの原因となる flatten_path_list を完全に廃止し、
        # 1層ずつ作った path_list を直接 GCode オブジェクトに流し込みます。
        gcode = gc.GCode(path_list)
        
        print("[5/5] G-code ファイルを書き出し中...")
        
        # 開始・終了G-codeの読み込み
        gcode.start_gcode("/Users/oosiroseigo/ゼミ/卒研用ディレクトリ/start_gcode.txt")
        gcode.end_gcode("/Users/oosiroseigo/ゼミ/卒研用ディレクトリ/end_gcode.txt")
        
        save_dir = "/Users/oosiroseigo/ゼミ/卒研用ディレクトリ/Gcode保存"
        os.makedirs(save_dir, exist_ok=True)
        gcode.save(os.path.join(save_dir, "gcode.gcode"))
        
        print("★すべての処理が完全に、ノーエラーで成功しました！")
        
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")

