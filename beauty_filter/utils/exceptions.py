class BeautyFilterError(Exception):
    """Base exception for beauty filter"""
    pass


class CameraError(BeautyFilterError):
    """Camera related errors"""
    pass


class MediaPipeError(BeautyFilterError):
    """MediaPipe processing errors"""
    pass


class ImageProcessingError(BeautyFilterError):
    """Image processing errors"""
    pass


class FileSystemError(BeautyFilterError):
    """File system related errors"""
    pass