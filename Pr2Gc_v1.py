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

def generate_3d_sdf_script(user_request: str) -> dict:
    print("[1/6] Gemini API にリクエストを送信中（汎用3D形状関数の生成）...")
    
    system_prompt = (
        "You are an expert 3D graphics programmer. Based on the user's request, "
        "write a Python function body that defines whether a given point (x, y, z) is inside the 3D object. "
        "Output strict JSON conforming to the response schema."
    )
    
    response_schema = gl.Schema(
        type=gl.Type.OBJECT,
        properties={
            "script": gl.Schema(
                type=gl.Type.OBJECT,
                properties={
                    "is_inside_code": gl.Schema(
                        type=gl.Type.STRING, 
                        description="x, y, z (単一の数値)を受け取り、その点が物体の『内側ならTrue、外側ならFalse』を返すPythonの関数内のロジックコード。改行は\\nにする。"
                    ),
                    "max_z": gl.Schema(type=gl.Type.NUMBER, description="このオブジェクト全体の最大の高さ(mm)")
                },
                required=["is_inside_code", "max_z"]
            )
        },
        required=["script"]
    )
    
    model = genai.GenerativeModel(
        model_name='models/gemini-3.5-flash-lite', 
        system_instruction=system_prompt
    )
    
    user_prompt = f"""
    ユーザーの要求: {user_request}

    【Pythonコード（is_inside_code）の記述ルール】
    - 入力変数として `x`, `y`, `z` (ミリメートル単位の数値) が与えられます。
    - あなたの書いたコードの最後で、その座標が物体の内側であれば `result = True`、外側なら `result = False` を代入してください。
    - `np` (numpy) の各種関数を自由に使って、数式、条件分岐、ループなどを駆使してロジックを組み立ててください。
    - x, y, z の値は -180 から +180 の範囲で与えられます。
    - 立体全体がZ=0（地面）より上に位置するように、Z軸方向に持ち上げて（オフセットして）コードを作ってください。
    【ロジックの例（ねじれた四角柱の場合）】
    # 高さzに応じて角度をひねる
    theta = z * 0.05
    nx = x * np.cos(theta) - y * np.sin(theta)
    ny = x * np.sin(theta) + y * np.cos(theta)
    # ひねった座標系で一辺20mmの四角形の内側か判定
    if abs(nx) <= 10 and abs(ny) <= 10:
        result = True
    else:
        result = False
    """
    
    response = model.generate_content(
        user_prompt,
        generation_config={
            "response_mime_type": "application/json",
            "response_schema": response_schema,
            "temperature": 0.2
        }
    )
    
    print("[2/6] Gemini API からレスポンス（形状関数）を受信しました。")
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
        # 1. Geminiから形状判定ロジックを取得
        data = generate_3d_sdf_script(user_input)
        is_inside_code = data["script"]["is_inside_code"]
        max_z = float(data["script"]["max_z"])
        
        print(f"📄 Geminiが生成したアルゴリズム:\n{is_inside_code}")
        print(f"📏 最大高さ: {max_z}mm")
        print("--------------------------------------------------")
        print("[4/6] Python側汎用スライサーで、空間をスキャンして3Dパスを計算中...")
        
        # 動的生成されたコードを関数化する
        def check_inside(x, y, z):
            local_vars = {"x": x, "y": y, "z": z, "np": np, "result": False}
            exec(is_inside_code, globals(), local_vars)
            return local_vars["result"]
        
        all_x = []
        all_y = []
        all_z = []
        
        # 1層あたりの角度走査点数（パスの滑らかさ）
        NUM_POINTS = 360
        angles = np.linspace(0, 2 * np.pi, NUM_POINTS)
        
        # Z軸方向に下から順にレイヤーをスライス
        current_z = LAYER_HEIGHT
        while current_z <= max_z:
            z = current_z
            layer_x = []
            layer_y = []
            
            # 各角度（全方位）に向かって、物体の「境界（外壁）」をレイ・キャスティング（探索）する
            for theta in angles:
                cos_t = np.cos(theta)
                sin_t = np.sin(theta)
                
                # 中心から外側に向かって、境界線をバイナリサーチ（二分探索）で高速・高精度に特定する
                low, high = 0.0, 150.0  # 探索範囲(mm)
                boundary_r = 0.0
                found = False
                
                for _ in range(10):  # 10回の二分探索で十分な精度（0.1mm以下）が出る
                    mid = (low + high) / 2.0
                    px = mid * cos_t
                    py = mid * sin_t
                    
                    if check_inside(px, py, z):
                        boundary_r = mid
                        low = mid  # もっと外側にあるか探す
                        found = True
                    else:
                        high = mid # 行き過ぎなので内側を探す
                        
                if found and boundary_r > 0.1:
                    layer_x.append(boundary_r * cos_t)
                    layer_y.append(boundary_r * sin_t)
            
            # 形状が存在したレイヤーをパスに追加
            if layer_x:
                layer_x.append(layer_x[0])  # 始点と結ぶ
                layer_y.append(layer_y[0])
                
                all_x.extend(layer_x)
                all_y.extend(layer_y)
                all_z.extend([z] * len(layer_x))
                
            current_z += LAYER_HEIGHT
            
        x_coords = np.array(all_x)
        y_coords = np.array(all_y)
        z_coords = np.array(all_z)
        
        if len(x_coords) == 0:
            raise ValueError("形状が検出されませんでした。Geminiの判定ロジックまたは原点位置を見直してください。")
            
        print(f"総データ数: {len(x_coords)} 点（総レイヤー数: {int(max_z/LAYER_HEIGHT)} 層）")
        
        print("[5/6] 3Dオブジェクトのパスを生成中...")
        path = gc.Path(x_coords, y_coords, z_coords)
        path.apply_default_settings()
        
        flat_paths = gc.path_generator.flatten_path_list([path])
        gcode = gc.GCode(flat_paths)
        
        print("[6/6] G-code ファイルを書き出し中...")
        gcode.start_gcode("/Users/oosiroseigo/ゼミ/卒研用ディレクトリ/start_gcode.txt")
        gcode.end_gcode("/Users/oosiroseigo/ゼミ/卒研用ディレクトリ/end_gcode.txt")
        
        save_dir = "/Users/oosiroseigo/ゼミ/卒研用ディレクトリ/Gcode保存"
        os.makedirs(save_dir, exist_ok=True)
        gcode.save(os.path.join(save_dir, "gcode.gcode"))
        
        print("★すべての処理が正常に完了しました！あらゆる指示に対応した造形を確認してください。")
        
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
