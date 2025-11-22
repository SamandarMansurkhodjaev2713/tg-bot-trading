# Create a completely clean version by rewriting the problematic section
with open('simple_forex_bot.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the aitrader_command function and rewrite the problematic message
lines = content.split('\n')

# Find the start of the function
func_start = None
for i, line in enumerate(lines):
    if 'async def aitrader_command' in line:
        func_start = i
        break

if func_start:
    # Find the problematic line and replace the entire message block
    for i in range(func_start, len(lines)):
        if 'Целевая точность' in lines[i]:
            # Replace this line and the surrounding context
            lines[i] = '                    "🎯 *Целевая точность: 92%+"*,'
            break

# Write back
with open('simple_forex_bot.py', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print("Fixed the message formatting issue")