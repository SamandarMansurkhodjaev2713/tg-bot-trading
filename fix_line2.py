# Create a fixed version of the file
with open('simple_forex_bot.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Fix line 132 (index 131) - add the missing closing quote
if 'Целевая точность' in lines[131]:
    lines[131] = '                    "🎯 *Целевая точность: 92%+"*,' + '\n'

# Write back to file
with open('simple_forex_bot.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("Fixed line 132 with proper quote closing")