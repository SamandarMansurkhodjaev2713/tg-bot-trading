# Check the exact content
with open('simple_forex_bot.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    
# Print line 132 (index 131)
print("Line 132:", repr(lines[131]))
print("Length:", len(lines[131]))
print("Characters:")
for i, char in enumerate(lines[131]):
    print(f"{i}: {repr(char)}")