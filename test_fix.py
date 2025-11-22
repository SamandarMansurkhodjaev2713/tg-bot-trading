# Test the string issue
line = '                    "🎯 *Целевая точность: 92%+\"*",'
print("Original:", repr(line))
print("Fixed:", repr(line.replace('\"', '"')))