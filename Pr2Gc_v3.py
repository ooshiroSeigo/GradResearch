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

def generate_3d_parametric_equations(user_request: str) -> dict:
    print("[1/5] Gemini API にリクエストを送信中（自由幾何学コードの取得）...")
    
    system_prompt = (
        "You are an expert 3D mathematical artist. Convert the user's request into a set of 3D parametric equations using Python. "
        "The arrays x and y must form closed loops for each layer. "
        "Output strict JSON matching the response schema."
    )
    
    response_schema = gl.Schema(
        type=gl.Type.OBJECT,
        properties={
            "parametric": gl.Schema(
                type=gl.Type.OBJECT,
                properties={
                    "x_expr": gl.Schema(type=gl.Type.STRING, description="変数tとzを用いた、X座標のNumPy一括計算式。例: 円柱なら '20 * np.cos(t)'"),
                    "y_expr": gl.Schema(type=gl.Type.STRING, description="変数tとzを用いた、Y座標のNumPy一括計算式。例: 円柱なら '20 * np.sin(t)'"),
                    "max_z": gl.Schema(type=gl.Type.NUMBER, description="オブジェクト全体の最大の高さ(mm)"),
                },
                required=["x_expr", "y_expr", "max_z"]
            )
        },
        required=["parametric"]
    )
    
    model = genai.GenerativeModel(
        model_name='models/gemini-3.5-flash-lite', 
        system_instruction=system_prompt
    )
    
    user_prompt = f"""
    ユーザーの要求: {user_request}

    【数式表現の基本ルール】
    - numpy(np)を使用してPythonで一括計算可能な数式（文字列）を出してください。
    - 1次元配列 `t` (0 から 2*np.pi まで) と現在の高さ `z` (数値) を用いて、X座標とY座標を決定してください。
    - 【最重要】各層の輪郭を完全に閉じるため、必ず数式内に `np.cos(t)` や `np.sin(t)` を組み込み、t=0 と t=2*np.pi の座標を一致させてください。
    - オブジェクトの最下部が Z = 0 に接地するようにしてください。

    【高度な幾何学表現のテクニック（これらを組み合わせて複雑な形を作ること）】
    1. ねじり・ひねり（Twist）:
       - 高さ `z` に比例して `t` の位相をずらす。
       - 例: `t_twisted = t + z * 0.1` と定義し、`np.cos(t_twisted)` のように使う。
    
    2. 縦方向の波打ち（Ripple/Wave）:
       - 高さ `z` に応じて半径を周期的に変化させる。
       - 例: `r = 15 + 3 * np.sin(z * 0.5)`
    
    3. 横方向の多角形・星型（Polygon/Star）:
       - `t` の高周波成分を半径に掛けることで、角（かど）や星型を作る。
       - 例（5角形風）: `r = 20 + 2 * np.cos(5 * t)`
       - 例（ギザギザの歯車）: `r = 15 + 1 * np.sign(np.sin(12 * t))`
    
    4. 形状のモーフィング（Morphing）:
       - 高さ `z` の割合（0〜1）に応じて、異なる2つの形状を線形補間する。
       - 例（底面が四角風、上面が円）: 
         `ratio = z / max_z`
         `r_square = 15 / np.maximum(np.abs(np.cos(t)), np.abs(np.sin(t)))`
         `r_circle = 15`
         `r = (1 - ratio) * r_square + ratio * r_circle`

    【様々な幾何学表現の記述例】
    - 例1（波打つ花瓶 - ツイスト＆リプル）:
      x_expr = "(15 + 3 * np.sin(z * 0.4)) * np.cos(t + z * 0.05)"
      y_expr = "(15 + 3 * np.sin(z * 0.4)) * np.sin(t + z * 0.05)"
      max_z = 50.0

    - 例2（多面体風のねじれタワー - 5角形ベース）:
      x_expr = "(20 + 4 * np.cos(5 * (t + z * 0.02))) * np.cos(t + z * 0.02)"
      y_expr = "(20 + 4 * np.cos(5 * (t + z * 0.02))) * np.sin(t + z * 0.02)"
      max_z = 60.0

    - 例3（数式による条件分岐 - 持ち手つきカップ）:
      # 高さzが特定の範囲だけ半径を外に広げて持ち手を作る（np.whereの活用）
      x_expr = "np.where((z > 15) & (z < 35) & (t > np.pi*0.8) & (t < np.pi*1.2), (15 + 10 * np.sin(t))*np.cos(t), 15 * np.cos(t))"
      y_expr = "np.where((z > 15) & (z < 35) & (t > np.pi*0.8) & (t < np.pi*1.2), (15 + 10 * np.sin(t))*np.sin(t), 15 * np.sin(t))"
      max_z = 40.0

    【3Dプリンターのための制約】
    - オーバーハング（空中への張り出し）が急激すぎると印刷に失敗します。高さ `z` が 1mm 増えるごとに、半径の変化（膨らみ）は最大でも 0.5mm 程度に収まるように数式を設計してください（傾斜角約45度以内）。
    - 1つの層の中で、ノズルが極端に激しく往復するような不連続な数式は避け、滑らかな連続関数（シグモイド、サイン、コサイン等）を優先してください。
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
        # 1. Geminiからパラメトリック数式データを取得
        data = generate_3d_parametric_equations(user_input)
        x_expr = data["parametric"]["x_expr"]
        y_expr = data["parametric"]["y_expr"]
        max_z = float(data["parametric"]["max_z"])
        
        print(f"[2/5] Geminiが生成した数式:\n  X = {x_expr}\n  Y = {y_expr}")
        print(f"📏 最大高さ: {max_z}mm")
        print("[3/5] 1次元NumPy配列で1層ずつ独立したPathを生成中...")
        
        full_object = []
        
        # Z軸方向に下から順に1層ずつPath（レイヤー）を生成
        current_z = LAYER_HEIGHT
        while current_z <= max_z:
            z = current_z
            
            # 0 から 2*np.pi までを分割。最初の点と最後の点が重なり、輪郭が綺麗に閉じる！
            t = np.linspace(0, 2 * np.pi, 200)
            
            #　evalを使って文字列の数式を実行し、X座標とY座標の配列を生成
            eval_env = {"t": t, "z": z, "np": np}
            layer_x = eval(x_expr, {"__builtins__": None}, eval_env)
            layer_y = eval(y_expr, {"__builtins__": None}, eval_env)
            layer_z = np.full_like(layer_x, z)
            
            # Pathオブジェクトを作成
            wall = gc.Path(layer_x, layer_y, layer_z)
            
            # 外周の壁を生成（厚みを追加）
            outer_wall = gc.Transform.offset(wall, 0.3)
            
            # リストに追加
            full_object.append(wall)
            full_object.append(outer_wall)
            
            current_z += LAYER_HEIGHT
            
        if len(full_object) == 0:
            raise ValueError("形状が検出されませんでした。数式を見直してください。")
            
        print(f"[4/5] gcoordinator パス生成中... (総Path数: {len(full_object)})")
        
        # 印刷順に並んだPathのリストを、そのままGCodeクラスに投入
        gcode = gc.GCode(full_object)
        
        print("[5/5] G-code ファイルを書き出し中...")
        gcode.start_gcode("/Users/oosiroseigo/ゼミ/卒研用ディレクトリ/start_gcode.txt")
        gcode.end_gcode("/Users/oosiroseigo/ゼミ/卒研用ディレクトリ/end_gcode.txt")
        
        save_dir = "/Users/oosiroseigo/ゼミ/卒研用ディレクトリ/Gcode保存"
        os.makedirs(save_dir, exist_ok=True)
        gcode.save(os.path.join(save_dir, "gcode.gcode"))
        
        print("成功しました！")
        
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
