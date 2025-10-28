# simple_test.py
print("🔍 بدء الاختبار البسيط...")

# 1. استيراد الحزمة أولاً
try:
    import ai_model_sentinel as sentinel
    print("✅ تم استيراد الحزمة بنجاح!")
    
    # 2. عرض ما بداخل الحزمة
    print("\n📋 محتويات الحزمة:")
    for item in dir(sentinel):
        if not item.startswith('_'):
            print(f"   🎯 {item}")
    
    # 3. حاول استخدام الوظائف الأساسية
    print("\n🧪 اختبار الوظائف:")
    if hasattr(sentinel, 'PromptGuard'):
        guard = sentinel.PromptGuard()
        result = guard.analyze("مرحباً هذا اختبار")
        print(f"   ✅ PromptGuard يعمل: {result}")
    else:
        print("   ❌ PromptGuard غير موجود")
        
except ImportError as e:
    print(f"❌ خطأ في الاستيراد: {e}")
    
    # تحقق من الحزم المثبتة
    import pkg_resources
    packages = [pkg.key for pkg in pkg_resources.working_set]
    print("\n🔍 الحزم المثبتة:")
    for pkg in packages:
        print(f"   {pkg}")