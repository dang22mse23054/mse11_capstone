import SearchBar, { IDispatchToProps, IStateToProps } from './SearchBar';
import { connect } from 'react-redux';
import { CategoryService } from 'servDir';
import { ICategory } from 'interfaceDir';

function mapStateToProps(store): IStateToProps {
	const videoReducer = store.videoReducer;
	const { initCategoryList: categoryList } = store.categoryReducer;
	console.log(categoryList)
	const listPage = videoReducer.list;
	// const errorMsg = reqInfo.errMsg;
	return {
		isLoading: listPage.isLoading,
		categoryList: categoryList.map(item => ({
			key: `${item.id}${item.name}`,
			value: item
		})),
	};
}


function mapDispatchToProps(dispatch, ownProps): IDispatchToProps {
	const dpToProps: IDispatchToProps = {
		
	};

	return dpToProps;
}

export default connect(mapStateToProps, mapDispatchToProps, null, { forwardRef: true })(SearchBar);