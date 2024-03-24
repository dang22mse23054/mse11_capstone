export class VideoInput {

	id?: number
	title?: string
	path?: string
	isEnabled: boolean;
	category: Array<number>;
	updatedAt?: string

	constructor(option = {}) {
		this.id = option.id != null ? Number(option.id) : undefined;
		this.title = option.title;
		this.string = option.string;
		this.isEnabled = option.isEnabled;
		this.category = option.category ? option.category.map(item => Number(item)) : [];
		this.updatedAt = option.updatedAt;
	}
}

export class VideoStatusInput {
	id: number;
	status: number;
	updatedAt: string;

	constructor(id, status, updatedAt) {
		this.id = id != null ? Number(id) : undefined;
		this.status = status != null ? Number(status) : undefined;
		this.updatedAt = updatedAt;
	}
}
