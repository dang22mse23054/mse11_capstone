export interface IPagingObj {
	totalItems?: number
	totalPages?: number
	currentPage?: number
}


export interface IGraphqlNode {
	__typename: string
}

export interface IGraphqlEdge<T extends IGraphqlNode> {
	node: T
	cursor: string
}

export interface IGraphqlPageInfo {
	total?: number
	limit?: number
	remaining?: number
	hasNext?: boolean
	remainingAfter?: number
	next?: string
	hasPrevious?: boolean
	remainingBefore?: number
	previous?: string
	currentPage?: number
	// using for reload page after CRUD
	lastCursor?: string
}

export interface IGraphqlPagingObj<T extends IGraphqlNode> {
	edges: Array<IGraphqlEdge<T>>
	pageInfo: IGraphqlPageInfo
}

export interface ICursorInput {
	cursor?: string
	nextCursor?: string
	prevCursor?: string
}