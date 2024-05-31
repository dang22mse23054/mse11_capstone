import React, { Component } from 'react';
import initWrapper from 'compDir/Wrapper';
import VideoList from 'compDir/pages/Video/List';
const VideoListWrapper = initWrapper(VideoList);

interface IState {
}

interface IProps {
}

class VideoPage extends Component<IProps, IState> {
	// Set default properties's values
	public static defaultProps: IProps = {
	}

	// Set default state
	public state: IState = {
	}

	public render(): React.ReactNode {
		return <VideoListWrapper {...this.props} />;
	}
}

export default VideoPage;