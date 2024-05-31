export class VideoInput {

	id?: number
	title?: string
	refFileName?: string;
	refFilePath?: string;
	isEnabled: boolean;
	categories: Array<number>;
	updatedAt?: string

	constructor(option = {}) {
		this.id = option.id != null ? Number(option.id) : undefined;
		this.title = option.title;
		this.refFileName = option.refFileName;
		this.refFilePath = option.refFilePath;
		this.isEnabled = option.isEnabled;
		this.categoryIds = option.categoryIds;
		this.updatedAt = option.updatedAt;
	}
}
