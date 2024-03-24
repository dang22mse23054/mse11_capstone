import Page, { IDispatchToProps, IStateToProps } from './Page';
import { connect } from 'react-redux';
import { Actions } from 'servDir/redux/actions';
import { Utils, VideoService, CategoryService } from 'servDir';
import { Paging } from 'constDir';
import { toast } from 'material-react-toastify';


function mapStateToProps(store): IStateToProps {
	const videoReducer = store.videoReducer;
	
	const listPage = videoReducer.list;

	return {
		videoList: listPage.videoList,
		pageInfo: listPage.pageInfo,
		isLoading: listPage.isLoading,
		holidayList: []
	};
}


function mapDispatchToProps(dispatch, ownProps): IDispatchToProps {
	let dpToProps: IDispatchToProps;
	

	return dpToProps = {
		initData: async (_component: Page) => {

			// === Init search options === //
			dispatch(CategoryService.initCategoryList());


			_component.setState({ firstLoading: false }, () => {
				
				const searchOpts = {
					..._component.state.searchBarRef.current?.state.searchOpts,
					keyword: '',
				};

				// === Search video === //
				dpToProps.doSearch({
					options: _component.prepareSearchOptions(searchOpts),
					limit: Paging.rowPerPage.default
				});

				_component.state.searchBarRef.current?.setState({ searchOpts });
			});
		},

		// reload video after update
		doReloadVideo: (videoId: number) => {
			dispatch(async (dispatch, getState) => {
				const videoReducer = getState().videoReducer as IVideoReducer;
				const listPage = videoReducer.list;
				const { currentPage, limit } = listPage.pageInfo;
				const options = listPage.searchInput.options;

				const cursor = { cursor: listPage.reloadCursorMap.get(currentPage) };
				const data = await VideoService.searchVideos(options, cursor, limit);
				// show large list
				if (data) {
					dispatch(Actions.VideoAction.ListPage.showSearchResult({ options, cursor }, data.list, data.pageInfo));
				}
			});
		},

		// options is IVideoSearchOpt type
		doSearch: async ({ options, cursor, limit }) => {
			dispatch(Actions.VideoAction.ListPage.setLoading());

			const data = await VideoService.searchVideos(options, cursor, limit);
			// show large list
			if (data) {
				return dispatch(Actions.VideoAction.ListPage.showSearchResult({ options, cursor }, data.list, data.pageInfo));
			}
			dispatch(Actions.VideoAction.ListPage.setLoading(false));
			toast.error('Cannot get Data');
		},

		doPaging: (pageNum: number) => {
			dispatch((dispatch, getState) => {
				const videoReducer = getState().videoReducer as IVideoReducer;
				const listPage = videoReducer.list;
				const { currentPage, next: nextCursor, previous: prevCursor, limit } = listPage.pageInfo;

				let cursor = null;
				if (pageNum == currentPage) {
					cursor = { cursor: listPage.reloadCursorMap.get(currentPage) };

				} else if (pageNum < currentPage) {
					cursor = { prevCursor };

				} else {
					cursor = { nextCursor };
				}

				dpToProps.doSearch({
					options: listPage.searchInput.options,
					cursor,
					limit
				});
			});
		},

		updateLimit: async (limit) => {
			dispatch((dispatch, getState) => {
				const videoReducer = getState().videoReducer;
				const listPage = videoReducer.list;
				dpToProps.doSearch({
					options: listPage.searchInput.options,
					limit
				});
			});

		},
	};
}

export default connect(mapStateToProps, mapDispatchToProps, null, { forwardRef: true })(Page);