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
    print("[1/5] Gemini API にリクエストを送信中（マルチオブジェクト対応）...")
    
    system_prompt = (
        "You are an expert 3D mathematical artist. Convert the user's request into 3D parametric equations using Python. "
        "Support single or multiple separate objects (islands) per layer. Each island must have its own closed loop."
        "Output strict JSON matching the response schema."
    )
    
    # 複数の島（オブジェクト）を配列(ARRAY)で受け取れるようにスキーマを定義
    response_schema = gl.Schema(
        type=gl.Type.OBJECT,
        properties={
            "max_z": gl.Schema(type=gl.Type.NUMBER, description="オブジェクト全体の最大の高さ(mm)"),
            "islands": gl.Schema(
                type=gl.Type.ARRAY,
                description="各独立したオブジェクト（島）の数式リスト",
                items=gl.Schema(
                    type=gl.Type.OBJECT,
                    properties={
                        "x_expr": gl.Schema(type=gl.Type.STRING, description="変数tとzを用いた、この島のX座標のNumPy計算式"),
                        "y_expr": gl.Schema(type=gl.Type.STRING, description="変数tとzを用いた、この島のY座標のNumPy計算式"),
                    },
                    required=["x_expr", "y_expr"]
                )
            )
        },
        required=["max_z", "islands"]
    )
    
    model = genai.GenerativeModel(
        model_name='models/gemini-3.5-flash-lite', 
        system_instruction=system_prompt
    )
    
    user_prompt = f"""
    ユーザーの要求: {user_request}

    【数式表現のルール】
    - numpy(np)を使用してpythonで計算可能な形で数式を出してください。
    - 1次元配列 `t` (0から2*np.piまで) と現在の高さ `z` (数値) を用いて、X座標とY座標を決定するPythonの数式（文字列）を作ってください。
    - 各島（オブジェクト）の輪郭を完全に閉じるため、必ず数式内に `np.cos(t)` や `np.sin(t)` を組み込んでください。
    - ユーザーが複数のオブジェクト（例: 「2つの円柱」「チェスの駒と台座」など）を求めた場合、それらを `islands` リストの中に別々の要素として分けて出力してください。
    - オブジェクトの最下部が Z = 0 に接地するようにしてください。
    - x_expr と y_expr は、各島ごとに独立した数式である必要があります。
    【記述例】
    - 例：離れた2つの円柱を配置する場合
      islands = [
        {{
          "x_expr": "-20 + 5 * np.cos(t)",  # 左側に配置された円柱
          "y_expr": "5 * np.sin(t)"
        }},
        {{
          "x_expr": "20 + 5 * np.cos(t)",   # 右側に配置された円柱
          "y_expr": "5 * np.sin(t)"
        }}
      ]
      max_z = 30.0
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
    # （ユーザー入力や設定読み込み部分はそのまま）
    user_input = input("作製したい3Dオブジェクトの形状や特徴を入力してください:\n> ")

    if user_input == "":
        print("❌ 入力が空です。処理を中断します。")
        exit()
    try:
        # 1. Geminiからデータを取得
        data = generate_3d_parametric_equations(user_input)
        islands = data["islands"]
        max_z = float(data["max_z"])
        
        print(f"[2/5] Geminiが生成したオブジェクト数: {len(islands)} 個")
        print(f"📏 最大高さ: {max_z}mm")
        print("[3/5] 各島ごとに独立したPathを生成中...")
        
        full_object = []
        
        current_z = LAYER_HEIGHT
        while current_z <= max_z:
            z = current_z
            t = np.linspace(0, 2 * np.pi, 200)
            eval_env = {"t": t, "z": z, "np": np}
            
            # 1つのレイヤー（高さz）の中で、登録されている島を順番に処理
            for island_idx, island in enumerate(islands):
                x_expr = island["x_expr"]
                y_expr = island["y_expr"]
                
                # 文字列の数式を実行
                layer_x = eval(x_expr, {"__builtins__": None}, eval_env)
                layer_y = eval(y_expr, {"__builtins__": None}, eval_env)
                layer_z = np.full_like(layer_x, z)
                
                # この島のための独立したPathを作成
                wall = gc.Path(layer_x, layer_y, layer_z)
                
                # 外周の壁を生成
                outer_wall = gc.Transform.offset(wall, 0.3)
                
                # リストに追加（独立したPathとして並べる）
                full_object.append(wall)
                full_object.append(outer_wall)
                
            current_z += LAYER_HEIGHT
            
        if len(full_object) == 0:
            raise ValueError("形状が検出されませんでした。")

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
        # （以降のG-code保存処理はそのまま）
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")