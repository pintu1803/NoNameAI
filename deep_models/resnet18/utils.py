

class Logger:

    @staticmethod
    def section(title):
        print("\n")
        print("=" * 80)
        print(f" {title}")
        print("=" * 80)

    @staticmethod
    def step(msg):
        print(f"  ➜ {msg}")

    @staticmethod
    def success(msg):
        print(f"  ✅ {msg}")

    @staticmethod
    def error(msg):
        print(f"  ❌ {msg}")