import Page, { IDispatchToProps, IStateToProps } from './Page';
import { connect } from 'react-redux';
import { Actions } from 'servDir/redux/actions';
import { Utils, TaskService } from 'servDir';
import { Paging } from 'constDir';
import { toast } from 'material-react-toastify';


function mapStateToProps(store): IStateToProps {
	return {
		holidayList: []
	};
}


function mapDispatchToProps(dispatch, ownProps): IDispatchToProps {
	let dpToProps: IDispatchToProps;

	return dpToProps = {
		initData: async (_component: Page) => {

			// === Init search options === //


			_component.setState({ firstLoading: false }, () => {

			});
		},

		
	};
}

export default connect(mapStateToProps, mapDispatchToProps, null, { forwardRef: true })(Page);