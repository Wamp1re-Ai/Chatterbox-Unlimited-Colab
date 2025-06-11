#!/usr/bin/env python3
"""
Fix syntax issues in the ChatterBox TTS notebook
Specifically fixes f-string escaping issues in JSON format
"""

import json
import re

def fix_notebook_syntax(notebook_path):
    """Fix syntax issues in Jupyter notebook"""
    print(f"🔧 Fixing syntax issues in {notebook_path}")
    
    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    fixes_applied = 0
    
    # Process each cell
    for cell in notebook.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            
            # Fix each line in the source
            for i, line in enumerate(source):
                original_line = line
                
                # Fix 1: Replace f"..." with f'...' to avoid escaping issues
                # Look for f-strings with double quotes that contain variables
                if 'f"' in line and '{' in line and '}' in line:
                    # Replace f"..." with f'...' but be careful about nested quotes
                    line = re.sub(r'f"([^"]*\{[^}]*\}[^"]*)"', r"f'\1'", line)
                
                # Fix 2: Fix specific timeout error message
                if 'TimeoutError(f\\' in line:
                    line = line.replace('f\\"', "f'").replace('\\")', "')")
                
                # Fix 3: Fix any remaining escaped quotes in f-strings
                if 'f\\' in line and '{' in line:
                    line = re.sub(r'f\\"([^"]*)"', r"f'\1'", line)
                
                # Update the line if it was changed
                if line != original_line:
                    source[i] = line
                    fixes_applied += 1
                    print(f"  Fixed line: {original_line.strip()[:50]}...")
    
    # Write the fixed notebook
    backup_path = notebook_path.replace('.ipynb', '_backup.ipynb')
    
    # Create backup
    with open(backup_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    
    # Write fixed version
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    
    print(f"✅ Applied {fixes_applied} fixes to {notebook_path}")
    print(f"📁 Backup saved as {backup_path}")
    
    return fixes_applied

def main():
    """Main function to fix notebook syntax"""
    print("🔧 ChatterBox TTS Notebook Syntax Fixer")
    print("=" * 50)
    
    notebook_files = [
        "ChatterBox_TTS_Fixed_CUDA.ipynb",
        "ChatterBox_Unlimited_Colab.ipynb"
    ]
    
    total_fixes = 0
    
    for notebook_file in notebook_files:
        try:
            fixes = fix_notebook_syntax(notebook_file)
            total_fixes += fixes
        except FileNotFoundError:
            print(f"⚠️ File not found: {notebook_file}")
        except Exception as e:
            print(f"❌ Error fixing {notebook_file}: {e}")
    
    print(f"\n🎉 Total fixes applied: {total_fixes}")
    print("\n💡 Key fixes:")
    print("- Fixed f-string escaping issues")
    print("- Replaced f\"...\" with f'...' to avoid JSON escaping")
    print("- Fixed TimeoutError message formatting")
    print("- Resolved syntax errors in notebook cells")

if __name__ == "__main__":
    main()
