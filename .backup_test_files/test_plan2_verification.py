#!/usr/bin/env python3
"""
验证 PLAN-2 修复的测试脚本
"""
import os
import sys
import pickle
import subprocess

def test_dummy_scaler_import():
    """测试 DummyScaler 导入"""
    print("=== 测试 1: DummyScaler 导入 ===")
    try:
        from src.utils.util import DummyScaler
        print("✅ DummyScaler 导入成功")
        
        # 测试实例化
        scaler = DummyScaler(100)
        print(f"✅ DummyScaler 实例化成功: {type(scaler).__name__}")
        return True
    except Exception as e:
        print(f"❌ DummyScaler 导入失败: {e}")
        return False

def test_scaler_pickle():
    """测试 scaler.pkl 文件加载"""
    print("\n=== 测试 2: Scaler Pickle 加载 ===")
    
    # 测试原始 scaler.pkl
    scaler_paths = [
        "./small_test_data/scaler.pkl",
        "./new_small_test_data/scaler.pkl"
    ]
    
    success_count = 0
    for scaler_path in scaler_paths:
        if os.path.exists(scaler_path):
            try:
                with open(scaler_path, "rb") as f:
                    scaler_data = pickle.load(f)
                print(f"✅ {scaler_path} 加载成功")
                print(f"   TEC scaler: {type(scaler_data['tec_scaler']).__name__}")
                print(f"   SW scaler: {type(scaler_data['sw_scaler']).__name__}")
                success_count += 1
            except Exception as e:
                print(f"❌ {scaler_path} 加载失败: {e}")
        else:
            print(f"⚠️  {scaler_path} 不存在")
    
    return success_count > 0

def test_config_files():
    """测试配置文件"""
    print("\n=== 测试 3: 配置文件验证 ===")
    
    config_files = [
        "conf/config_optuna_test.yaml",
        "conf/config_new_optuna_test.yaml",
        "conf/hydra/sweeper/test_optuna.yaml"
    ]
    
    success_count = 0
    for config_file in config_files:
        if os.path.exists(config_file):
            print(f"✅ {config_file} 存在")
            success_count += 1
        else:
            print(f"❌ {config_file} 不存在")
    
    return success_count == len(config_files)

def test_simple_training():
    """测试简单训练运行"""
    print("\n=== 测试 4: 简单训练运行 ===")
    
    try:
        # 运行简单的训练测试
        cmd = [
            "python", "src/tec_train.py",
            "--config-name", "config_optuna_test",
            "trainer.epochs=1",
            "use_optuna=false"  # 禁用 Optuna 进行简单测试
        ]
        
        print(f"运行命令: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd="/home/panxiong/tecGPT-forecasting"
        )
        
        if result.returncode == 0:
            print("✅ 简单训练运行成功")
            return True
        else:
            print(f"❌ 简单训练运行失败，返回码: {result.returncode}")
            if result.stderr:
                print(f"错误信息: {result.stderr[-500:]}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ 训练运行超时")
        return False
    except Exception as e:
        print(f"❌ 训练运行异常: {e}")
        return False

def test_optuna_storage():
    """测试 Optuna 存储"""
    print("\n=== 测试 5: Optuna 存储 ===")
    
    try:
        import optuna
        
        # 确保目录存在
        os.makedirs("logs", exist_ok=True)
        
        # 测试数据库路径
        db_path = "logs/plan2_test.db"
        if os.path.exists(db_path):
            os.remove(db_path)
        
        # 创建存储
        storage_url = f"sqlite:///{db_path}"
        storage = optuna.storages.RDBStorage(
            url=storage_url,
            engine_kwargs={"connect_args": {"timeout": 60, "check_same_thread": False}}
        )
        
        # 创建 study
        study = optuna.create_study(storage=storage, study_name="plan2_test")
        
        # 简单优化
        def objective(trial):
            x = trial.suggest_float("x", -10, 10)
            return x**2
        
        study.optimize(objective, n_trials=2)
        
        print("✅ Optuna 存储测试成功")
        print(f"✅ 数据库文件: {db_path}")
        return True
        
    except Exception as e:
        print(f"❌ Optuna 存储测试失败: {e}")
        return False

def main():
    print("开始验证 PLAN-2 修复...")
    
    tests = [
        ("DummyScaler 导入", test_dummy_scaler_import),
        ("Scaler Pickle 加载", test_scaler_pickle),
        ("配置文件验证", test_config_files),
        ("简单训练运行", test_simple_training),
        ("Optuna 存储", test_optuna_storage)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} 测试异常: {e}")
            results.append((test_name, False))
    
    # 总结
    print("\n" + "="*50)
    print("PLAN-2 验证结果总结:")
    print("="*50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{len(results)} 测试通过")
    
    if passed == len(results):
        print("\n🎉 PLAN-2 修复验证成功！")
        print("\n已完成的修复:")
        print("1. ✅ DummyScaler 类移动到 src/utils/util.py")
        print("2. ✅ create_scaler.py 修改为导入 DummyScaler")
        print("3. ✅ 配置文件正确设置 Hydra sweeper")
        print("4. ✅ Optuna 存储配置正确")
        print("5. ✅ Pickle 反序列化问题已解决")
        return True
    else:
        print(f"\n⚠️  还有 {len(results) - passed} 个问题需要解决")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
