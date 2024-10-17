import os
import time

#---------------------------------------------------------------
# 엠파워 문서명, 문서내용을 리스트로 입력받아서 서버에 파일로 저장하는 함수
# -in : folder_path : 저장할 폴더 경로.
# -in : rfile_name_list : 엠파워 문서 rfile_name 목록들.(*리스트로 입력)
# -in : rfile_text_list : 엠파워 문서 내용들.(*리스트로 입력)
# -out: response['rfile_path_list'] : 저장된 파일 목록들.(*리스트로 출력)
#---------------------------------------------------------------
def mpower_save_docs(folder_path:str, rfile_name_list:list, rfile_text_list:list):
    
    file_path_list:list = []
    
    assert folder_path, f"folder_path is empty!"
    assert len(rfile_name_list) > 0, f"rfile_name_list is empty"
    assert len(rfile_text_list) > 0, f"rfile_text_list is empty"
    
    try:
        # 폴더가 없으며 새로 생성.
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # rfile_name으로 파일 생성하고 rfile_text 내용을 write 함
        for rfile_name, rfile_text in zip(rfile_name_list, rfile_text_list):
            file_path = folder_path + "/" + rfile_name

            # text 파일로 저장 => 기존 파일이 있으면 덮어씀.
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(rfile_text)
            
            file_path_list.append(file_path)
    
    except Exception as e:
            msg = f'*mpower_save_doc is Fail!!..(error: {e})'
            #myutils.log_message(f'[info][/extract] {msg}')

    response = {"file_path_list":file_path_list}
    return response