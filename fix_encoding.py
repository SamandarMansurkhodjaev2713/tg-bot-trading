# Read the file with proper encoding
with open('simple_forex_bot.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the problematic line and fix it
lines = content.split('\n')
for i, line in enumerate(lines):
    if 'Целевая точность: 92%+' in line:
        # Replace with a properly formatted line
        lines[i] = '                    "🎯 *Целевая точность: 92%+"*,'
        print(f"Fixed line {i+1}")
        break

# Write back
with open('simple_forex_bot.py', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print("File encoding fixed")