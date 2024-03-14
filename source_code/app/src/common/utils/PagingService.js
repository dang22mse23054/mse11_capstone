const PagingService  = {

	async doPaging(pageNum, totalRecords, getDataCallback) {
		const DEFAULT_RECORDS_PER_PAGE = parseInt(process.env.DEFAULT_RECORDS_PER_PAGE ? process.env.DEFAULT_RECORDS_PER_PAGE : 10);

		if (totalRecords == null || totalRecords == 0) {return null;}
		
		let totalPages = parseInt(totalRecords / DEFAULT_RECORDS_PER_PAGE);
		totalPages = totalRecords % DEFAULT_RECORDS_PER_PAGE === 0 ? totalPages : totalPages + 1;

		//check page input
		if (pageNum < 1 || pageNum > totalPages) {
			return null;
		}

		let limitIndex = DEFAULT_RECORDS_PER_PAGE;
		if (pageNum == totalPages) {
			limitIndex = totalPages == 1 ? totalRecords : DEFAULT_RECORDS_PER_PAGE;
		}
		let offsetIndex = pageNum == 1 ? 0 : (pageNum - 1) * DEFAULT_RECORDS_PER_PAGE;
        
		return { 
			totalPages, 
			totalRecords, 
			recordsPerPage: DEFAULT_RECORDS_PER_PAGE, 
			rawData: await getDataCallback(limitIndex, offsetIndex)
		};
	}
};

module.exports = PagingService;