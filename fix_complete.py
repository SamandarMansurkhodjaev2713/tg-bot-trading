# Create a completely fixed version
with open('simple_forex_bot.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find and replace the problematic line
lines = content.split('\n')
for i, line in enumerate(lines):
    if 'Целевая точность' in line and '92%+"*' in line:
        # Replace with properly formatted line
        lines[i] = '                    "🎯 *Целевая точность: 92%+"*,'
        print(f"Fixed line {i+1}: {repr(lines[i])}")
        break

# Write back
with open('simple_forex_bot.py', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print("File fixed")