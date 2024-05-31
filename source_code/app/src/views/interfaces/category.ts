export interface ICategory {
	id?: number
	name?: string

	/**
	 * Male => 0, 
	 * Female => 1
	 * 
	*/
	gender?:number

	/**
	 * CHILDREN (00~12) => 0, 
	 * TEENAGERS (13~17) => 1, 
	 * ADULT (18~44) => 2,
	 * MIDDLE_AGED (45~60) => 3, 
	 * ELDERLY (61~12) => 4
	*/
	age?:number
}
