import Setting, { IDispatchToProps, IStateToProps, IProps } from './Setting';
import { connect } from 'react-redux';
import { Actions } from 'servDir/redux/actions';
import { AuthService, CategoryService } from 'servDir';
import { ICategory, ICategoryOption } from 'interfaceDir';
import { CheckTypes, Status } from 'constDir';

function mapStateToProps(store): IStateToProps {
	const videoReducer = store.videoReducer;
	const { initCategoryList: categoryList } = store.categoryReducer;
	return {
		videoInfo: videoReducer.setting,
		categoryList: categoryList.map(item => ({
			key: `${item.id}${item.name}`,
			value: item
		})),
	};
}

function mapDispatchToProps(dispatch, ownProps: IProps): IDispatchToProps {
	const dpToProps: IDispatchToProps = {
		initData: async (_component: Page) => {
			AuthService.checkSession(true)
				.then(async ({ error, authData }) => {

				})
				.catch((e) => {
					console.error(e);
				});
		},

		changeTitle: (title) => {
			dispatch(Actions.VideoAction.SettingPage.changeTitle(title));
		},

		changeCategories: (chips, reason) => {
			console.log(chips)
			dispatch(Actions.VideoAction.SettingPage.changeCategories(chips.filter(item => item.value.status != Status.DELETED)));
		},

		changeRefFile: (fileInfo) => {
			const { status } = fileInfo;

			if (status == Status.DELETED) {
				fileInfo.fileName = undefined;
				fileInfo.filePath = undefined;
			}
			dispatch(Actions.VideoAction.SettingPage.changeRefFile(fileInfo));
		},

	};
	return dpToProps;
}

export default connect(mapStateToProps, mapDispatchToProps)(Setting);