import { useState, useEffect, useCallback } from 'react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Plus, Trash2 } from 'lucide-react'
import { Dialog, DialogContent, DialogTrigger, DialogHeader, DialogTitle, DialogDescription } from '../ui/dialog'

export type ModelItem = {
  name: string
  type: 'text' | 'image' | 'video'
}

interface ModelsListProps {
  models: Record<string, { type?: 'text' | 'image' | 'video' }>
  onChange: (
    models: Record<string, { type?: 'text' | 'image' | 'video' }>
  ) => void
  label?: string
}

export default function AddModelsList({
  models,
  onChange,
  label = 'Models',
}: ModelsListProps) {
  const [modelItems, setModelItems] = useState<ModelItem[]>([])
  const [isInitialized, setIsInitialized] = useState(false)
  const [newModelName, setNewModelName] = useState('')
  const [openAddModelDialog, setOpenAddModelDialog] = useState(false)

  // Initialize state only once when models prop changes from outside
  useEffect(() => {
    if (!isInitialized) {
      const items = Object.entries(models).map(([name, config]) => ({
        name,
        type: (config.type || 'text') as 'text' | 'image' | 'video',
      }))
      setModelItems(items.length > 0 ? items : [])
      setIsInitialized(true)
    }
  }, [models, isInitialized])

  const notifyChange = useCallback(
    (items: ModelItem[]) => {
      // Filter out empty model names and convert back to object format
      const validModels = items.filter((model) => model.name.trim())
      const modelsConfig: Record<
        string,
        { type?: 'text' | 'image' | 'video' }
      > = {}

      validModels.forEach((model) => {
        modelsConfig[model.name] = { type: model.type }
      })

      onChange(modelsConfig)
    },
    [onChange]
  )

  const handleAddModel = () => {
    if (newModelName) {
      const newItems = [
        ...modelItems,
        { name: newModelName, type: 'text' as const },
      ]
      setModelItems(newItems)
      notifyChange(newItems)
      setNewModelName('')
      setOpenAddModelDialog(false)
    }
  }

  const handleRemoveModel = (index: number) => {
    if (modelItems.length > 1) {
      const newItems = modelItems.filter((_, i) => i !== index)
      setModelItems(newItems)
      notifyChange(newItems)
    }
  }

  const handleModelChange = (
    index: number,
    field: keyof ModelItem,
    value: string
  ) => {
    const newItems = [...modelItems]
    if (field === 'type') {
      newItems[index][field] = value as 'text' | 'image' | 'video'
    } else {
      newItems[index][field] = value
    }
    setModelItems(newItems)
    notifyChange(newItems)
  }

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <Label>{label}</Label>
        <Dialog open={openAddModelDialog} onOpenChange={setOpenAddModelDialog}>
          <DialogTrigger asChild>
            <Button variant="secondary" size="sm">
              <Plus className="h-4 w-4 mr-1" />
              Add Model
            </Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Add a new model</DialogTitle>
              <DialogDescription>
                Enter the name of the model you want to add.
              </DialogDescription>
            </DialogHeader>
            <div className="space-y-5">
              <Label>Model Name</Label>
              <Input
                type="text"
                placeholder="openai/gpt-4o"
                value={newModelName}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    handleAddModel()
                  }
                }}
                onChange={(e) => setNewModelName(e.target.value)}
              />
              <Button type="button" onClick={handleAddModel} className="w-full">
                Add Model
              </Button>
            </div>
          </DialogContent>
        </Dialog>
      </div>

      <div className="space-y-2">
        {modelItems.map((model, index) => (
          <div key={index} className="flex items-center justify-between">
            <p className="w-[50%]">{model.name}</p>
            <div className="flex items-center gap-2">
              <Select
                value={model.type}
                onValueChange={(value) =>
                  handleModelChange(index, 'type', value)
                }
              >
                <SelectTrigger className="w-24">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="text">text</SelectItem>
                  <SelectItem value="image">image</SelectItem>
                  <SelectItem value="video">video</SelectItem>
                </SelectContent>
              </Select>
              {modelItems.length > 1 && (
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={() => handleRemoveModel(index)}
                  className="h-10 w-10 p-0"
                >
                  <Trash2 className="h-4 w-4" />
                </Button>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
