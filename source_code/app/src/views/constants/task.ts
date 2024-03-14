export const ExportModes = {
	DeadlineDate: { label: '締切期限', value: 1, isDefault: true},
	FinishedDate: { label: 'タスク完了日', value: 2 },
} as const;

export const ExportModeValues = [
	ExportModes.DeadlineDate.value, 
	ExportModes.FinishedDate.value
] as const;