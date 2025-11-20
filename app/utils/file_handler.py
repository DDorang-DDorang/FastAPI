"""파일 업로드 및 처리 유틸리티"""
import os
from fastapi import UploadFile
from pydub import AudioSegment


def get_unique_filepath(base_dir: str, base_name: str, ext: str) -> str:
    """
    동일한 파일명이 존재하면 _1, _2, _3 ... 식으로 이름 변경
    """
    candidate = os.path.join(base_dir, f"{base_name}{ext}")
    counter = 1
    while os.path.exists(candidate):
        candidate = os.path.join(base_dir, f"{base_name}_{counter}{ext}")
        counter += 1
    return os.path.abspath(candidate)


def save_upload_file(upload_file: UploadFile, base_name: str, upload_dir: str) -> str:
    """
    파일 저장 (중복 시 이름 자동 변경)
    base_name에 확장자가 포함되어 있으면 추가하지 않음
    """
    upload_ext = os.path.splitext(upload_file.filename)[1].lower()
    base_root, base_ext = os.path.splitext(base_name)

    # base_name에 이미 확장자가 있으면 그대로, 없으면 upload_file에서 확장자 사용
    ext = base_ext if base_ext else upload_ext

    save_path = get_unique_filepath(upload_dir, base_root, ext)
    with open(save_path, "wb") as f:
        f.write(upload_file.file.read())
    os.chmod(save_path, 0o666)
    return os.path.abspath(save_path)


def merge_chunks(original_filename: str, total_chunks: int, ext: str, upload_dir: str) -> str:
    """
    조각난 파일 합치기 (기본 이름: original_filename, 중복 시 _1, _2 등 추가)
    """
    base_name = os.path.splitext(original_filename)[0]
    merged_path = get_unique_filepath(upload_dir, base_name, ext)

    with open(merged_path, "wb") as merged:
        for i in range(total_chunks):
            part_path = os.path.join(upload_dir, f"{original_filename}_chunk_{i}{ext}")
            with open(part_path, "rb") as part:
                merged.write(part.read())
            os.remove(part_path)
    os.chmod(merged_path, 0o666)
    return os.path.abspath(merged_path)


def convert_to_wav(file_path: str) -> str:
    """mp4 -> wav 변환, wav는 그대로 반환"""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".mp4":
        wav_path = file_path.rsplit(".", 1)[0] + ".wav"
        audio = AudioSegment.from_file(file_path, format="mp4")
        audio.export(wav_path, format="wav")
        return wav_path
    elif ext == ".wav":
        return file_path
    else:
        raise ValueError("지원되지 않는 파일 형식입니다. wav 또는 mp4만 허용됩니다.")

