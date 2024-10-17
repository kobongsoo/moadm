import random
import shutil
import os

# 새로운 파일을 생성하는 함수 
# => 원본파일에서 줄단위로 나누어서 %remove_per만큼 줄을 제거하고, %shuffle_per만큼 줄을 섞어서 원본과 유사한 파일을 생성하는 함수 
# -in: srcFilePath : 원본 파일
# -in: tgtFilePath : 새롭게 만들 파일
# -in: remove_per : 원본파일에서 제거할 줄 % 
# -in: shuffle_per : 원본파일에서 섞을 줄 %
def create_similar_doc(srcFilePath:str, tgtFilePath:str, remove_per:int=10, shuffle_per:int=10):
    
    assert srcFilePath, f"srcFilePath is Empty!!"
    assert tgtFilePath, f"tgtFilePath is Empty!!"
    assert remove_per > 1, f"remove_per must be greater than 1!!"
    assert shuffle_per > 1, f"shuffle_per must be greater than 1 1!!"
    
    msg:str = "success"
    #print(f'*shuffle_per:{shuffle_per}\n*remove_per:{remove_per}')
    
    try:
        # src 파일을 읽어온다.
        with open(srcFilePath, 'r', encoding='utf-8') as file:
            lines = file.readlines()
    except Exception as e:
        msg = f'readlines is Fail!!..(error: {e})'
        return 1001, msg
        
    # 전체 줄 수의 10% 계산합니다.
    total_lines = len(lines)
    remove_count = int(total_lines * (remove_per/100)) # 제거할 줄의 수
    shuffle_count = int(total_lines * (shuffle_per/100))     # 섞을 줄의 수
    
    #print(f'*total_lines:{total_lines}\n*remove_count:{remove_count}\n*shuffle_count:{shuffle_count}')
    
    # 제거할 줄을 선택합니다.
    remove_indices = random.sample(range(total_lines), remove_count)
    lines = [line for i, line in enumerate(lines) if i not in remove_indices]

    # 남은 줄들 중에서 섞을 줄을 선택합니다.
    shuffle_indices = random.sample(range(len(lines)), shuffle_count)
    
    # 선택된 줄을 섞습니다.
    shuffled_lines = [lines[i] for i in shuffle_indices]
    random.shuffle(shuffled_lines)

    # 섞인 줄을 원래 위치에 다시 넣습니다.
    for i, index in enumerate(shuffle_indices):
        lines[index] = shuffled_lines[i]
        
    # 새로운 디렉토리가 존재하지 않으면 생성합니다.
    tgt_dir = os.path.dirname(tgtFilePath)
    if not os.path.exists(tgt_dir):
        os.makedirs(tgt_dir)
     
    try:
        # 섞은 내용을 새로운 파일로 저장합니다.
        with open(tgtFilePath, 'w', encoding='utf-8') as file:
            file.writelines(lines)
    except Exception as e:
        msg = f'writelines is Fail!!..(error: {e})'
        return 1002, msg
    
    return 0, msg

#로컬파일 삭제
def delete_local_file(filepath:str):
    if len(filepath) < 1:
        return
    
    try:
        os.remove(filepath)
        print(f"File deleted: {filepath}")
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        
# 디렉토이 및 하위 디렉토리 모든 파일 삭제하는 합수
def delete_all_contents(directory):
    # 디렉토리 내의 모든 파일 및 디렉토리 탐색
    for root, dirs, files in os.walk(directory, topdown=False):
        for file in files:
            # 파일 경로 생성
            file_path = os.path.join(root, file)
            try:
                # 파일 삭제
                os.remove(file_path)
                print(f"Deleted file: {file_path}")
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
        for dir in dirs:
            # 디렉토리 경로 생성
            dir_path = os.path.join(root, dir)
            try:
                # 디렉토리 삭제
                shutil.rmtree(dir_path)
                print(f"Deleted directory: {dir_path}")
            except Exception as e:
                print(f"Failed to delete {dir_path}. Reason: {e}")
