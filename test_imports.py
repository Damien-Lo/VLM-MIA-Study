#!/usr/bin/env python3

import sys
import os

# Add the minigpt directory to the path
sys.path.append('./minigpt')

def test_imports():
    """Test if the key imports work correctly."""
    try:
        print("Testing MiniGPT imports...")
        
        # Test registry import
        from minigpt.minigpt4.common.registry import registry
        print("✓ Registry import successful")
        
        # Test conversation imports
        from minigpt.minigpt4.conversation.interact import CONV_VISION_Vicuna0, CONV_VISION_LLama2
        print("✓ Conversation imports successful")
        
        # Test model imports
        from minigpt.minigpt4.models.base_model import BaseModel
        print("✓ Base model import successful")
        
        # Test src.model import
        from src.model import generate
        print("✓ src.model.generate import successful")
        
        print("\n🎉 All imports successful!")
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
