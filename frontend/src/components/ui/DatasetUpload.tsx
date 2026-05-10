import { useRef, useState } from 'react'
import { UploadCloud, CheckCircle, XCircle, Loader } from 'lucide-react'
import { api } from '../../services/api'

interface Props {
  onUploadSuccess?: () => void
}

type UploadState = 'idle' | 'uploading' | 'success' | 'error'

export default function DatasetUpload({ onUploadSuccess }: Props) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [uploadState, setUploadState] = useState<UploadState>('idle')
  const [message, setMessage] = useState<string>('')
  const [fileName, setFileName] = useState<string>('')

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    const ext = file.name.split('.').pop()?.toLowerCase()
    if (ext !== 'csv' && ext !== 'tsv') {
      setUploadState('error')
      setMessage('Solo se aceptan archivos CSV.')
      return
    }

    setFileName(file.name)
    setUploadState('uploading')
    setMessage('')

    try {
      const result = await api.uploadDataset(file)
      if (result?.message) {
        setUploadState('success')
        setMessage(result.message)
        onUploadSuccess?.()
      } else if (result?.detail) {
        setUploadState('error')
        setMessage(result.detail)
      } else {
        setUploadState('success')
        setMessage('Dataset cargado. Haz clic en "Entrenar Modelos" para continuar.')
        onUploadSuccess?.()
      }
    } catch (err: any) {
      setUploadState('error')
      setMessage(err.message ?? 'Error al subir el archivo.')
    } finally {
      // Reset file input so the same file can be re-uploaded
      if (inputRef.current) inputRef.current.value = ''
    }
  }

  const triggerInput = () => inputRef.current?.click()

  return (
    <div className="w-full">
      <input
        ref={inputRef}
        type="file"
        accept=".csv,.tsv"
        onChange={handleFileChange}
        className="hidden"
      />

      <button
        onClick={triggerInput}
        disabled={uploadState === 'uploading'}
        className="flex items-center gap-2 px-5 py-2.5 bg-white border-2 border-dashed border-slate-300 text-slate-600 rounded-xl hover:border-blue-400 hover:text-blue-600 hover:bg-blue-50 transition-all disabled:opacity-50 disabled:cursor-not-allowed w-full justify-center"
      >
        {uploadState === 'uploading' ? (
          <Loader size={18} className="animate-spin text-blue-600" />
        ) : (
          <UploadCloud size={18} />
        )}
        <span className="text-sm font-medium">
          {uploadState === 'uploading'
            ? `Subiendo ${fileName}…`
            : 'Subir Dataset (CSV)'}
        </span>
      </button>

      {uploadState === 'success' && (
        <div className="flex items-start gap-2 mt-2 text-green-700 bg-green-50 border border-green-200 rounded-lg px-3 py-2 text-xs">
          <CheckCircle size={14} className="mt-0.5 flex-shrink-0" />
          <span>{message}</span>
        </div>
      )}

      {uploadState === 'error' && (
        <div className="flex items-start gap-2 mt-2 text-red-700 bg-red-50 border border-red-200 rounded-lg px-3 py-2 text-xs">
          <XCircle size={14} className="mt-0.5 flex-shrink-0" />
          <span>{message}</span>
        </div>
      )}
    </div>
  )
}
