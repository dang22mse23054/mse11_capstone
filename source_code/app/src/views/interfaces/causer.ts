
export interface ICAUser {
	id: number,
	uid: string,
	fullname: string,
	normalize: string,
	kana: string,
	email: string,
	division2: string,
	division3: string,
	division4: string,
	division5: string,
	division6: string,
	status?: number
	isDeleted?: boolean
}