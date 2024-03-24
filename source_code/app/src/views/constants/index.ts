export * from './paging';
export * from './video';

export const TIME_ZONE = 'Asia/Tokyo';

export const DateFmt = {
	YYYYMMDD: 'YYYY-MM-DD',
};

export const Status = {
	SKIP: 0,
	NEW: 1,
	UPDATED: 2,
	DELETED: 3,
};
export const StatusMap = { 1: Status.NEW, 2: Status.UPDATED, 3: Status.DELETED, 0: Status.SKIP };

