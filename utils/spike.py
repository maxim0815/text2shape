import language_tool_python

tool = language_tool_python.LanguageTool('en-US')
text = 'A sentence with a error in the Hitchhiker’s Guide tot he Galaxy'

while True:
    tool.correct(text)