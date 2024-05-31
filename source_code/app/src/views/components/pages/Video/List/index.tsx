import Page, { IDispatchToProps, IStateToProps } from './Page';
import { connect } from 'react-redux';
import { Actions } from 'servDir/redux/actions';
import { Utils, VideoService, CategoryService, Validator } from 'servDir';
import { Paging } from 'constDir';
import { toast } from 'material-react-toastify';
import { VideoInput } from 'inputModelDir';


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

		loadSetting: async (modalTypeId: number, videoId: number) => {
			const video: IVideo = await VideoService.convertVideo(videoId > 0 ? await VideoService.getVideo(Number(videoId)) : {});
			return dispatch(Actions.VideoAction.SettingPage.setVideoInfo(video));
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

		changeEnabled: async (video: any, callback, options: any = {}) => {
			dispatch(async (dispatch, getState) => {
				const { list } = getState().videoReducer as IVideoReducer;
				const { pageInfo } = list;

				await VideoService.updateVideoStatus(video)
					.then(result => {
						if (result) {
							// === Search video === //
							dpToProps.doPaging(pageInfo.currentPage);
						}
						return result;
					})
					.then(callback);
			});
		},

		deleteVideo: async (video: any, callback, options: any = {}) => {
			dispatch(async (dispatch, getState) => {
				const { list } = getState().videoReducer as IVideoReducer;
				const { pageInfo } = list;

				await VideoService.updateVideoStatus(video, true)
					.then(result => {
						if (result) {
							// === Search video === //
							dpToProps.doPaging(pageInfo.currentPage);
						}
						return result;
					})
					.then(callback);
			});
		},

		onSubmitVideo: async () => {
			return dispatch(async (dispatch, getState) => {
				const { setting, list } = getState().videoReducer as IVideoReducer;
				const { pageInfo } = list;
				const video = {
					...setting,
				};

				return VideoService.insertOrUpdateVideo(new VideoInput(video))
					.then(result => {
						if (result) {
							// === Search video === //
							dpToProps.doPaging(pageInfo.currentPage);
						}
						return result;
					});
			});
		},

		validateVideo: async () => {
			return dispatch(async (dispatch, getState) => {
				const { setting } = getState().videoReducer as IVideoReducer;

				const video: IVideoSettingAO = {
					...setting,
				};
				const result = Validator.isValidVideo(video);
				dispatch(Actions.VideoAction.SettingPage.setVideoError(result));
				return result == true;
			});
		},
	};
}

export default connect(mapStateToProps, mapDispatchToProps, null, { forwardRef: true })(Page);