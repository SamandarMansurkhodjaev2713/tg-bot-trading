# Read the current file and replace the broken function
with open('simple_forex_bot.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Read the working function
with open('final_function.py', 'r', encoding='utf-8') as f:
    new_function = f.read()

# Find the function boundaries and replace
start_marker = '    async def aitrader_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):'
end_marker = '        except Exception as e:'

# Split the content
parts = content.split(start_marker)
before_part = parts[0]

if len(parts) > 1:
    after_parts = parts[1].split(end_marker)
    after_part = end_marker + after_parts[1] if len(after_parts) > 1 else ''
else:
    after_part = ''

# Add proper indentation to the new function
indented_function = '\n'.join(['    ' + line if line.strip() else line for line in new_function.split('\n')])

# Write the complete fixed file
with open('simple_forex_bot.py', 'w', encoding='utf-8') as f:
    f.write(before_part + indented_function + '\n' + after_part)

print("Successfully replaced the aitrader_command function with working version")