let g:vebugger_path_python = system('pipenv run which python')[:-2]
noremap ùs :VBGstartPDB -m atarieyes features train -e BreakoutDeterministic-v4<CR>
